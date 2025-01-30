from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_tokenize():
    """Load dataset and tokenizer"""
    dataset = load_dataset("openlifescienceai/medmcqa", split='train')
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    return dataset, tokenizer

def count_tokens(example, tokenizer):
    """Count tokens in the question"""
    tokens = tokenizer(example['question'], return_tensors="pt", truncation=False)
    example['token_count'] = len(tokens['input_ids'][0])
    return example

def split_question(question: str) -> tuple[str, str]:
    """Split question into context and formatted question"""
    sentences = question.split('. ')
    return ('. '.join(sentences[:-1]) + '.').strip(), sentences[-1].strip()

def transform_example(example):
    """Transform dataset example into desired structure"""
    option_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    correct_option = example['cop']
    
    if correct_option not in option_map:
        raise ValueError(f"Invalid correct option: {correct_option}")
    
    correct_answer = example[f'op{option_map[correct_option].lower()}']
    context, question_formatted = split_question(example['question'])
    
    data = {
        'Correct Answer': correct_answer,
        'Correct Option': option_map[correct_option],
        'Options': {
            'A': example['opa'],
            'B': example['opb'],
            'C': example['opc'],
            'D': example['opd']
        },
        'Question': example['question']
    }
    
    return {
        'id': example['id'],
        'data': data,
        'subject_name': example['subject_name'],
        'Context': context,
        'Question_formatted': question_formatted,
        'Options': data['Options'],
        'Final_answer': f"{option_map[correct_option]}. {correct_answer}"
    }

def filter_examples(example):
    """Filter examples based on criteria"""
    # Check for image tags in all relevant fields
    has_img_tag = any('<img' in str(value) for value in [
        example['question'],
        example['opa'],
        example['opb'],
        example['opc'],
        example['opd']
    ])
    
    # Check for empty fields after transformation
    transformed = transform_example(example)
    has_valid_fields = (
        transformed['Question_formatted'].strip() != '' and 
        transformed['Final_answer'].strip() != ''
    )
    
    # Keep example only if it has no image tags and valid fields
    return not has_img_tag and has_valid_fields

def process_dataset():
    """Main processing function"""
    # Load dataset and tokenizer
    dataset, tokenizer = load_and_tokenize()
    
    # Apply filtering before token counting to save processing time
    dataset = dataset.filter(filter_examples, num_proc=32)
    
    # Count tokens and filter by token count
    dataset = dataset.map(
        lambda x: count_tokens(x, tokenizer),
        num_proc=32
    )
    dataset = dataset.filter(lambda x: x['token_count'] > 100, num_proc=32)
    
    # Transform dataset structure
    dataset = dataset.map(transform_example, num_proc=32)
    
    # Remove unnecessary columns
    columns_to_remove = [
        'question', 'opa', 'opb', 'opc', 'opd', 'cop',
        'choice_type', 'exp', 'token_count'
    ]
    dataset = dataset.remove_columns(columns_to_remove)
    
    return dataset

transformed_dataset = process_dataset()

# Print results
print(f"Number of samples in transformed dataset: {len(transformed_dataset)}")
print("\nSample transformed data:")
print(transformed_dataset[0])

print("\nPushing to Hub")
transformed_dataset.push_to_hub("OpenMedical/medical-raw", "medmcqa")
