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

# Load and process the dataset
dataset = load_dataset("OpenMedical/m1-raw", "qwen7b-distil", split='train')
processed_dataset = process_dataset(dataset)
print(processed_dataset)