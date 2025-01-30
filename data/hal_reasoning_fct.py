from datasets import load_dataset
from transformers import AutoTokenizer
import ast

def load_and_tokenize():
    """Load dataset and tokenizer"""
    dataset = load_dataset("openlifescienceai/Med-HALT", "reasoning_FCT", split='train')
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

def get_letter_index(number: int) -> str:
    """Convert numeric index to letter index"""
    return chr(65 + number)  # 65 is ASCII for 'A'

def transform_options(example):
    """Transform options from 0/1/2/3 to A/B/C/D and remove 'correct answer'"""
    # Convert string representation of dictionary to actual dictionary
    options_dict = ast.literal_eval(example['options'])
    
    # Create new options dictionary with A/B/C/D keys
    number_to_letter = {'0': 'A', '1': 'B', '2': 'C', '3': 'D'}
    new_options = {
        number_to_letter[key]: value 
        for key, value in options_dict.items() 
        if key != 'correct answer'
    }
    
    # Update example with new options
    example['Options'] = new_options  # Store as dictionary
    
    # Create Final_answer column
    letter_index = get_letter_index(example['correct_index'])
    example['Final_answer'] = f"{letter_index}. {example['correct_answer']}"
    
    # Remove old options column
    del example['options']
    
    return example

def add_split_question_columns(example):
    """Add Context and Question_formatted columns"""
    context, question_formatted = split_question(example['question'])
    example['Context'] = context
    example['Question_formatted'] = question_formatted
    return example

def create_data_column(example):
    """Create the 'data' column with formatted information as a dictionary"""
    example['data'] = {
        "Correct Answer": example['correct_answer'],
        "Correct Option": get_letter_index(example['correct_index']),
        "Options": example['Options'],
        "Question": example['question']
    }
    return example

def filter_examples(example):
    """Filter examples based on image content and valid fields"""
    # Check for image-related keywords in relevant fields
    image_keywords = ['<img', 'image', 'picture']
    has_image_content = any(
        any(keyword in str(value).lower() for keyword in image_keywords)
        for value in [
            example['question'],
            str(example['Options'])  # Check in the Options dictionary
        ]
    )

    # Check for empty fields after transformation (ensure to do this after other transformations)
    has_valid_fields = (
        example['Question_formatted'].strip() != '' and
        example['Final_answer'].strip() != ''
    )

    return not has_image_content and has_valid_fields

def process_dataset():
    """Main processing function"""
    # Load dataset and tokenizer
    dataset, tokenizer = load_and_tokenize()
    
    # Count tokens and filter by token count
    dataset = dataset.map(
        lambda x: count_tokens(x, tokenizer),
        num_proc=32
    )
    dataset = dataset.filter(lambda x: x['token_count'] > 100, num_proc=32)
    
    # Transform options and add Final_answer
    dataset = dataset.map(
        transform_options,
        num_proc=32
    )
    
    # Add split question columns
    dataset = dataset.map(
        add_split_question_columns,
        num_proc=32
    )

    # Filter out examples with image content or invalid fields
    dataset = dataset.filter(
        filter_examples,
        num_proc=32
    )

    # Create the 'data' column (now as a dictionary)
    dataset = dataset.map(
        create_data_column,
        num_proc=32
    )

    # Select only the desired columns
    dataset = dataset.select_columns(
        ['id', 'subject_name', 'topic_name', 'data', 'Context', 'Question_formatted', 'Options', 'Final_answer']
    )

    return dataset

# Process the dataset
transformed_dataset = process_dataset()

# Print results
print(f"Number of samples in transformed dataset: {len(transformed_dataset)}")
print("\nSample transformed data:")
print(transformed_dataset)

# Print first example to verify changes
print("\nFirst example:")
print(transformed_dataset[0])

print("Pushing to Hub")
transformed_dataset.push_to_hub("OpenMedical/medical-raw", "hal_reasoning_fct")
