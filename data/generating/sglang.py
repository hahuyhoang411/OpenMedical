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

from filtering import extract_boxed_answer, compare_answer

file_lock = Lock()


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
            async with session.post(
                f"http://{args.api_addr}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": prompt}],
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


async def process_example(
    example, 
    session, 
    args, 
    output_file, 
    pbar,
    processed_info: ProcessedExample = None
):
    prompt = args.prompt_template.format(prompt=example[args.prompt_column])
    
    generations = processed_info.existing_generations if processed_info else []
    finish_reasons = processed_info.existing_finish_reasons if processed_info else []
    api_metadata = processed_info.existing_api_metadata if processed_info else []
    
    remaining_generations = args.num_generations - len(generations)
    max_retries = 3  # Maximum retries per generation
    
    if remaining_generations > 0:
        try:
            for _ in range(remaining_generations):
                valid_response = False
                retry_count = 0
                
                while not valid_response and retry_count < max_retries:
                    completion = await generate_completion(session, prompt, args)
                    
                    if completion is None:
                        print("Error getting completion")
                        break
                        
                    generation = completion["choices"][0]["message"]["content"]
                    extracted = extract_boxed_answer(generation)
                    is_valid = compare_answer(extracted, example['correct_option'])
                    
                    if is_valid:
                        valid_response = True
                        generations.append(generation)
                        finish_reasons.append(completion["choices"][0]["finish_reason"])
                        api_metadata.append(completion["usage"])
                    else:
                        # Save invalid response
                        invalid_result = {
                            "uuid": hashlib.md5(str(example[args.uuid_column]).encode()).hexdigest(),
                            
                            "prompt": prompt,
                            "generation": generation,
                            "extracted_answer": extracted,
                            "correct_option": example['correct_option'],
                            "retry_count": retry_count
                        }
                        
                        async with file_lock:
                            invalid_file = f"{os.path.splitext(args.output_file)[0]}_invalid.jsonl"
                            async with aiofiles.open(invalid_file, mode="a") as f:
                                await f.write(json.dumps(invalid_result) + "\n")
                                await f.flush()
                        
                        retry_count += 1
                        print(f"Invalid response format, retrying ({retry_count}/{max_retries})")
                
                if not valid_response:
                    print(f"Failed to get valid response after {max_retries} attempts")
                    raise Exception(f"Failed to get valid response after {max_retries} attempts")

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
        default=(
            "# Instructions:\n"
            "Task: You are a world-class Medical Doctor at Stanford. "
            "You have a 4.0 GPA at Harvard Medical School. Answer the given question correctly."
            "Let's think step by step.\n\n"
            "# OUTPUT FORMAT:\n"
            "Now, please output the final answer in a format:\n"
            "The final answer is: \\boxed{{<content of 'answer'>}}.\n"
            "# Question:\n"
            "{prompt}"
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-concurrent", type=int, default=1000)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name, split="train").shuffle()
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