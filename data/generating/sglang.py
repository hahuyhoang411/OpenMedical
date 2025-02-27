import argparse
import asyncio
import hashlib
import json
import os
import random
from asyncio import Lock
from typing import Set, Dict, Tuple
from dataclasses import dataclass

from datasets import load_dataset
from tqdm.asyncio import tqdm

import aiofiles
import aiohttp
import uvloop

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from filtering import extract_boxed_answer

file_lock = Lock()

# Default prompts - can be easily modified directly in the script

# For translation - fr
# DEFAULT_SYSTEM_PROMPT = """Vous êtes un expert en linguistique et en traduction avec de nombreuses années d’expérience.
# # Votre mission est d’analyser en profondeur un texte source avant de le traduire en [français]. L’objectif est d’assurer une traduction précise, contextuellement appropriée, tout en conservant le sens et le style du texte original. Suivez ces étapes :

# # 1. Identifier le sujet et le sens principal :
# # • Résumez brièvement le contenu essentiel du texte de manière claire et compréhensible.

# # 2. Analyser le public cible et le contexte :
# # • Déterminez à qui s’adresse le texte (ex. : experts, étudiants, consommateurs).
# # • Évaluez le contexte d’utilisation (ex. : académique, marketing, personnel).

# # 3. Analyser le style, le ton et l’émotion :
# # • Identifiez le registre du texte (ex. : formel, créatif, technique).
# # • Décrivez le ton et les émotions véhiculées (ex. : joyeux, sérieux, urgent) et leur impact sur le sens.

# # 4. Examiner le vocabulaire et les expressions spécifiques :
# # • Listez les mots ou expressions clés et expliquez leur signification dans le contexte.
# # • Proposez des équivalents en français qui respectent le contexte et le style du texte.

# # 5. Gérer les éléments spécifiques :
# # • Notez la manière d’aborder les termes techniques, les structures complexes ou les tournures particulières.
# # • Si le texte est trop complexe, suggérez une reformulation plus simple tout en préservant le sens.

# # 6. Anticiper les défis et proposer des solutions :
# • Identifiez les difficultés potentielles de traduction (ex. : différences culturelles, perte de sens figuré).
# • Suggérez des stratégies pour surmonter ces défis.

# 7. Évaluer la cohérence et la qualité :
# • Vérifiez la cohérence terminologique, le maintien des idées et du style dans la traduction.
# • Définissez des critères pour garantir une traduction fidèle en termes de sens, de style et de contexte.

# Traduisez la version anglaise suivante en français. Ne résolvez aucun problème, traduisez uniquement le texte.

# # Version anglaise:"""

# For adjust thinking - en
# DEFAULT_SYSTEM_PROMPT="You are a highly critical and analytical individual with a sharp, discerning personality, modeled after a seasoned critic—imagine a meticulous reviewer or a skeptical scholar. You excel at critical thinking and dissecting questions to reveal their deeper intent and underlying needs. Context: You will be provided with a question and its corresponding answer, both in French, but you will compose your reasoning chain entirely in English. Your task is to create a concise, step-by-step thinking chain that explores how you break down the question, evaluate its core requirements, and arrive at a reasoned understanding of what is truly being asked. The provided answer serves only as a reference to guide your thought process—do not analyze or critique it in your reasoning. Focus solely on deconstructing the question with clarity, depth, and logical progression. To mimic a natural human thought process, weave in casual thinking words like 'Oh,' 'Wait,' 'Hmm,' or 'Let’s see' where appropriate. Keep your tone sharp yet conversational."

# For classification - en
DEFAULT_SYSTEM_PROMPT= "You are an expert in question analysis with a sharp, precise, and analytical mind. Your task is to classify a given question into one of two categories: `reasoning` (requires logical deduction, problem-solving, or a chain of thought to answer) or `understanding` (tests factual knowledge or comprehension, requiring little to no reasoning). Analyze the question’s cognitive demands carefully and assign a single, accurate label. Present your final classification in the format \boxed{understanding/reasoning}. Each question only have 1 label either `understanding` or `reasoning`."

# For conclusion
# DEFAULT_SYSTEM_PROMPT="You are a thoughtful and reflective individual with a clear and methodical approach to reasoning. Your task is to carefully read the provided question and its accompanying thinking process, then craft a final conclusion that ties everything together for the reader. The conclusion should summarize all the correct steps from the thinking process in a way that’s easy to understand yet comprehensive enough to stand alone, since the reader cannot see the detailed thinking process itself. Focus on making the conclusion concise, accessible, and reflective of the key insights derived from your analysis, ensuring it fully captures the essence of your reasoning without requiring additional context. At the end of the conclusion, put your final answer within \boxed{}."

DEFAULT_USER_PROMPT_TEMPLATE = """{prompt}"""
USE_SYSTEM_PROMPT = True  # Set to False to use only user prompt


@dataclass
class ProcessedExample:
    uuid: str
    num_generations: int
    existing_generations: list
    existing_finish_reasons: list
    existing_api_metadata: list


async def generate_completion(session, prompt, args):
    retry_budget = 10
    while retry_budget > 0:
        try:
            await asyncio.sleep(random.uniform(0.0, 0.1))
            
            # Build messages based on whether to use system prompt
            messages = []
            if args.use_system_prompt:
                messages.append({"role": "system", "content": args.system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            async with session.post(
                f"http://{args.api_addr}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": messages,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                },
                headers={"Authorization": "Bearer EMPTY"},
            ) as response:
                return await response.json(content_type=None)
        except Exception as e:
            print(f"API error (will retry): {e}")
            retry_budget -= 1
            await asyncio.sleep(10)
    return None


async def generate_with_retry(example, session, prompt, args):
    """Generate a completion with validation and retry logic."""
    for attempt in range(args.max_retries):
        try:
            completion = await generate_completion(session, prompt, args)
            
            if completion is None:
                print(f"API returned None (attempt {attempt+1}/{args.max_retries})")
                continue
                
            generation = completion["choices"][0]["message"]["content"]
            finish_reason = completion["choices"][0]["finish_reason"]
            
            # Check for error conditions
            has_boxed_format = extract_boxed_answer(generation) is not None
            is_length_truncated = finish_reason == "length"
            
            # Apply validation based on args
            format_valid = not args.require_boxed_format or has_boxed_format
            not_truncated = not args.reject_truncated or finish_reason != "length"
            
            if format_valid and not_truncated:
                # Success - valid generation
                return {
                    "content": generation,
                    "finish_reason": finish_reason,
                    "api_metadata": completion["usage"]
                }
            else:
                # Log the error
                if not has_boxed_format and args.require_boxed_format:
                    print(f"Missing boxed format (attempt {attempt+1}/{args.max_retries})")
                if is_length_truncated and args.reject_truncated:
                    print(f"Response truncated (attempt {attempt+1}/{args.max_retries})")
                    
                # Save invalid response
                invalid_result = {
                    "uuid": hashlib.md5(str(example[args.uuid_column]).encode()).hexdigest(),
                    "prompt": prompt,
                    "generation": generation,
                    "retry_count": attempt
                }
                
                async with file_lock:
                    invalid_file = f"{os.path.splitext(args.output_file)[0]}_invalid.jsonl"
                    async with aiofiles.open(invalid_file, mode="a") as f:
                        await f.write(json.dumps(invalid_result) + "\n")
                        await f.flush()
                
                # Add a short delay before retry
                await asyncio.sleep(random.uniform(0.5, 2.0))
        
        except Exception as e:
            print(f"Error during generation (attempt {attempt+1}/{args.max_retries}): {e}")
            await asyncio.sleep(random.uniform(1.0, 5.0))
    
    # If we get here, all retries failed
    return None


async def process_example(
    example, 
    session, 
    args, 
    output_file, 
    pbar,
    processed_info: ProcessedExample = None
):
    # Check if the prompt column value is None
    if example[args.prompt_column] is None:
        # Create a result with None values for generations
        result = {
            **example,
            "generations": None,
            "finish_reasons": None,
            "api_metadata": None,
        }
        
        # Write to file with lock
        async with file_lock:
            async with aiofiles.open(output_file, mode="a") as f:
                await f.write(json.dumps(result) + "\n")
                await f.flush()
        
        pbar.update(1)
        return result
    
    prompt = args.prompt_template.format(prompt=example[args.prompt_column])
    
    generations = processed_info.existing_generations if processed_info else []
    finish_reasons = processed_info.existing_finish_reasons if processed_info else []
    api_metadata = processed_info.existing_api_metadata if processed_info else []
    
    remaining_generations = args.num_generations - len(generations)
    
    if remaining_generations > 0:
        try:
            # Create a task for each remaining generation
            raw_generation_results = []
            for _ in range(remaining_generations):
                gen_result = await generate_with_retry(example, session, prompt, args)
                if gen_result:
                    raw_generation_results.append(gen_result)
                else:
                    print(f"Failed to generate after {args.max_retries} retries")
                    # If we completely failed to generate, update with what we have
                    if generations:
                        pbar.update(1)
                        return {
                            **example,
                            "generations": generations,
                            "finish_reasons": finish_reasons,
                            "api_metadata": api_metadata,
                        }
                    else:
                        pbar.update(1)
                        return None

            # Add all successful generations
            for result in raw_generation_results:
                generations.append(result["content"])
                finish_reasons.append(result["finish_reason"])
                api_metadata.append(result["api_metadata"])

            # Combine original dataset fields with generations
            result = {
                **example,
                "generations": generations,
                "finish_reasons": finish_reasons,
                "api_metadata": api_metadata,
            }

            async with file_lock:
                if processed_info and args.continue_incomplete:
                    await update_existing_line(output_file, example[args.uuid_column], result)
                else:
                    async with aiofiles.open(output_file, mode="a") as f:
                        await f.write(json.dumps(result) + "\n")
                        await f.flush()

            pbar.set_postfix(active=len(pbar.active_tasks), refresh=False)
            pbar.update(1)
            return result

        except Exception as e:
            print(f"Error processing example: {e}")
            pbar.update(1)
            return None
    else:
        pbar.update(1)
        return None


async def update_existing_line(output_file, uuid, new_data):
    """Update an existing line in the output file."""
    temp_file = f"{output_file}.temp"
    async with aiofiles.open(output_file, mode="r") as f_in:
        async with aiofiles.open(temp_file, mode="w") as f_out:
            async for line in f_in:
                data = json.loads(line)
                if str(data[args.uuid_column]) == str(uuid):
                    await f_out.write(json.dumps(new_data) + "\n")
                else:
                    await f_out.write(line)
    
    os.replace(temp_file, output_file)


async def load_processed_examples(output_file, uuid_column) -> Dict[str, ProcessedExample]:
    processed_examples = {}
    if os.path.exists(output_file):
        async with aiofiles.open(output_file, mode="r") as f:
            async for line in f:
                try:
                    data = json.loads(line)
                    uuid = hashlib.md5(str(data[uuid_column]).encode()).hexdigest()
                    processed_examples[uuid] = ProcessedExample(
                        uuid=uuid,
                        num_generations=len(data.get("generations", [])),
                        existing_generations=data.get("generations", []),
                        existing_finish_reasons=data.get("finish_reasons", []),
                        existing_api_metadata=data.get("api_metadata", [])
                    )
                except json.JSONDecodeError:
                    continue
    return processed_examples


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--dataset-sub", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--prompt-column", type=str, required=True)
    parser.add_argument("--uuid-column", type=str, required=True)
    parser.add_argument("--api-addr", type=str, default="localhost:39876")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--continue-incomplete", action="store_true", 
                       help="Continue processing examples with fewer generations than requested")
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=DEFAULT_USER_PROMPT_TEMPLATE,
        help="Template for formatting user prompts"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt to use for chat completions"
    )
    parser.add_argument(
        "--use-system-prompt",
        action="store_true",
        default=USE_SYSTEM_PROMPT,
        help="Whether to use the system prompt (true) or user-only messages (false)"
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-concurrent", type=int, default=1000)
    
    # Add retry-related arguments
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed or invalid generations"
    )
    parser.add_argument(
        "--require-boxed-format",
        action="store_true",
        default=False,
        help="Require \\boxed{} format in the response"
    )
    parser.add_argument(
        "--reject-truncated",
        action="store_true",
        default=True,
        help="Reject and retry generations with 'length' finish reason"
    )
    
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name, args.dataset_sub, split="train")
    processed_examples = await load_processed_examples(args.output_file, args.uuid_column)
    
    # Calculate total work needed
    total_remaining = 0
    for example in dataset:
        uuid = hashlib.md5(str(example[args.uuid_column]).encode()).hexdigest()
        if uuid in processed_examples:
            if args.continue_incomplete:
                remaining = args.num_generations - processed_examples[uuid].num_generations
                if remaining > 0:
                    total_remaining += 1
        else:
            total_remaining += 1
    
    print(f"Found {len(processed_examples)} processed examples")
    if args.continue_incomplete:
        incomplete = sum(1 for info in processed_examples.values() 
                       if info.num_generations < args.num_generations)
        print(f"Found {incomplete} examples with incomplete generations")
    
    if not os.path.exists(args.output_file):
        async with aiofiles.open(args.output_file, mode="w") as f:
            await f.write("")

    active_tasks: Set[asyncio.Task] = set()

    pbar = tqdm(
        total=total_remaining,
        desc="Generating responses",
        unit="row",
        mininterval=2,
        smoothing=0.0001,
    )
    pbar.active_tasks = active_tasks

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=60 * 60),
        connector=aiohttp.TCPConnector(limit=args.max_concurrent, ttl_dns_cache=300, keepalive_timeout=60 * 60),
    ) as session:
        for example in dataset:
            uuid = hashlib.md5(str(example[args.uuid_column]).encode()).hexdigest()
            should_process = False
            processed_info = None
            
            if uuid in processed_examples:
                if args.continue_incomplete:
                    info = processed_examples[uuid]
                    if info.num_generations < args.num_generations:
                        should_process = True
                        processed_info = info
            else:
                should_process = True
            
            if should_process:
                # Wait if we've hit the concurrency limit
                while len(active_tasks) >= args.max_concurrent:
                    done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        try:
                            await task
                        except Exception as e:
                            print(f"Task failed: {e}")

                task = asyncio.create_task(
                    process_example(example, session, args, args.output_file, pbar, processed_info)
                )
                active_tasks.add(task)
                task.add_done_callback(active_tasks.discard)

                pbar.set_postfix(active=len(active_tasks), refresh=True)

        # Wait for remaining tasks
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)

    pbar.close()


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(main())
