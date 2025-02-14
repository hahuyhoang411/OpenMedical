import re
from datasets import load_dataset

import re
from datasets import load_dataset

def normalize_answer_format(answer):
    """
    Normalize different answer formats to a standard format 'A, B, C, D'
    Handles various formats and separators including special characters.
    """
    if answer is None:
        return None
        
    # Convert to uppercase and strip whitespace
    answer = answer.strip().upper()
    
    # If single character answer, return it
    if len(answer) == 1 and answer.isalpha():
        return answer
    
    # Step 1: Replace all non-alphanumeric characters with comma
    # This handles any special characters like ~!@#$%^&*()_+<>?:"{}|
    standardized = re.sub(r'[^A-Z0-9]+', ',', answer)
    
    # Step 2: Handle case where options are just concatenated (e.g., 'ABCD')
    if ',' not in standardized:
        # Split string into individual characters if they're all letters
        if all(c.isalpha() for c in standardized):
            standardized = ','.join(list(standardized))
    
    # Step 3: Clean up the options
    options = []
    for opt in standardized.split(','):
        opt = opt.strip()
        # Only keep valid options (single letters or numbers)
        if opt and (
            (len(opt) == 1 and opt.isalpha()) or  # Single letter
            opt.isdigit() or                      # Number
            (len(opt) <= 2 and opt.isalnum())     # Alphanumeric up to 2 chars (e.g., A1)
        ):
            options.append(opt)
    
    # Step 4: Remove duplicates while preserving order
    seen = set()
    options = [x for x in options if not (x in seen or seen.add(x))]
    
    # Step 5: Join with standard separator
    return ', '.join(options) if options else None

def extract_boxed_answer(text):
    """
    Extract the answer from \\boxed{...} pattern.
    Returns None if no match is found.
    """
    pattern = r'\\boxed{([^}]*)}'
    match = re.search(pattern, text)
    return match.group(1).strip() if match else None

def compare_answer(extracted, correct):
    """
    Compare extracted answer with correct answer.
    Both answers are normalized before comparison.
    """
    if extracted is None or correct is None:
        return False
    
    # Normalize both answers
    normalized_extracted = normalize_answer_format(extracted)
    normalized_correct = normalize_answer_format(correct)
    
    # Handle None cases after normalization
    if normalized_extracted is None or normalized_correct is None:
        return False
    
    return normalized_extracted == normalized_correct

def process_dataset(dataset):
    """
    Process the dataset and add correctness column for each generation
    """
    def process_example(example):
        correctness = []
        for generation in example['generations']:
            extracted_answer = extract_boxed_answer(generation)
            is_correct = compare_answer(extracted_answer, example['correct_option'])
            correctness.append(is_correct)
        
        example['correctness'] = correctness
        return example
    
    processed_dataset = dataset.map(process_example, num_proc=32)
    return processed_dataset

def count_true_values(correctness_list):
    # Ensure we only count up to 3 values
    if len(correctness_list) > 3:
        correctness_list = correctness_list[:3]
    
    # Count True values
    return sum(1 for x in correctness_list if x is True)


# Load dataset and process
qwen7b_ins_dataset = load_dataset("OpenMedical/m1-raw", "qwen7b-ins", split='train')
qwen7b_ins_dataset = process_dataset(dataset)
deepnous8b_dataset = load_dataset("OpenMedical/m1-raw", "deepnous8b", split='train')
deepnous8b_dataset = process_dataset(dataset)
qwen7b_distil_dataset = load_dataset("OpenMedical/m1-raw", "qwen7b-distil", split='train')
qwen7b_distil_dataset = process_dataset(dataset)

# Apply the function to create new column for qwen7b_distil_dataset
qwen7b_distil_dataset = qwen7b_distil_dataset.map(
    lambda x: {'correctness_count': count_true_values(x['correctness'])}
)

# Apply the function to create new column for qwen7b_ins_dataset
qwen7b_ins_dataset = qwen7b_ins_dataset.map(
    lambda x: {'correctness_count': count_true_values(x['correctness'])}
)

# Apply the function to create new column for qwen7b_ins_dataset
deepnous8b_dataset = deepnous8b_dataset.map(
    lambda x: {'correctness_count': count_true_values(x['correctness'])}
)

# Create dictionary to track questions that should be filtered out
questions_to_filter = set()

# Check all three datasets for perfect scores (3/3)
for idx in range(len(qwen7b_distil_dataset)):
    distil_example = qwen7b_distil_dataset[idx]
    ins_example = qwen7b_ins_dataset[idx]
    deepnous_example = deepnous8b_dataset[idx]
    
    # Generate UUID using the question as key
    uuid = hashlib.md5(str(distil_example['question']).encode()).hexdigest()
    
    # If any model got 3/3, add to filter set
    if (distil_example['correctness_count'] > 1 or 
        ins_example['correctness_count'] > 1 or 
        deepnous_example['correctness_count'] > 1):
        questions_to_filter.add(uuid)

# Filter function
def filter_questions(example):
    uuid = hashlib.md5(str(example['question']).encode()).hexdigest()
    return uuid not in questions_to_filter

# Apply filtering to get final dataset
filtered_dataset = qwen7b_distil_dataset.filter(filter_questions)

# Print statistics
print(f"Original dataset size: {len(qwen7b_distil_dataset)}")
print(f"Number of questions filtered out: {len(questions_to_filter)}")
print(f"Final dataset size: {len(filtered_dataset)}")


# Verify the filtering worked as expected by checking a few examples
print("\nVerification of first few examples:")
for idx in range(min(20, len(filtered_dataset))):
    example = filtered_dataset[idx]
    uuid = hashlib.md5(str(example['question']).encode()).hexdigest()
    scores = question_scores[uuid]
    
    print(f"\nQuestion {idx + 1}:")
    print(f"UUID: {uuid}")
    print(f"Question: {scores['question']}")
    print(f"Correctness counts:")
    print(f"- Distil: {scores['distil']}")
    print(f"- Ins: {scores['ins']}")
    print(f"- Deepnous: {scores['deepnous']}")
    
    # Additional verification
    assert scores['distil'] <= 1 and scores['ins'] <= 1 and scores['deepnous'] <= 1, \
        f"Found scores > 1: Distil={scores['distil']}, Ins={scores['ins']}, Deepnous={scores['deepnous']}"

# Filter columns

filtered_dataset = filtered_dataset.remove_columns(['generations', 'finish_reasons', 'api_metadata', 'correctness', 'correctness_count'])
print(filtered_dataset)
# filtered_dataset.push_to_hub("OpenMedical/medical-data-stage1")