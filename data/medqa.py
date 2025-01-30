from huggingface_hub import login
login(token=HF_KEY)

from datasets import concatenate_datasets, load_dataset

# Load the medqa dataset (both train and dev splits)
train_dataset = load_dataset("openlifescienceai/medqa", split='train')
dev_dataset = load_dataset("openlifescienceai/medqa", split='dev')

# Combine train and dev datasets
combined_dataset = concatenate_datasets([train_dataset,dev_dataset])

# Inspect the combined dataset
print(combined_dataset)
print(combined_dataset[0]['data']['Question'])

def split_question(question: str) -> tuple[str, str]:
    """
    Split a question into context and formatted question.
    The context contains all sentences except the last one,
    and the formatted question is the last sentence.
    
    Args:
        question (str): The input question text
        
    Returns:
        tuple[str, str]: (context, question_formatted)
    """
    sentences = question.split('. ')
    return ('. '.join(sentences[:-1]) + '.').strip(), sentences[-1].strip()

# Apply the function to create new columns
processed_dataset = combined_dataset.map(
    lambda x: {
        'Context': split_question(x['data']['Question'])[0],
        'Question_formatted': split_question(x['data']['Question'])[1],
        'Options': x['data']['Options'],
        'Final_answer': f"{x['data']['Correct Option']}. {x['data']['Correct Answer']}"
    },
    num_proc=32
)

# Verify the results
print("\nSample result:")
print("Context:", processed_dataset[0]['Context'])
print("Formatted Question:", processed_dataset[0]['Question_formatted'])
print("Options:", processed_dataset[0]['Options'])
print("Final Answer:", processed_dataset[0]['Final_answer'])

def is_valid_question(question: str) -> bool:
    """
    Check if a question string is valid.
    A valid question must:
    1. Not be empty
    2. End with a question mark
    3. Have more than just whitespace or punctuation
    
    Args:
        question (str): The question string to validate
        
    Returns:
        bool: True if the question is valid, False otherwise
    """
    if not isinstance(question, str):
        return False
        
    # Remove whitespace
    question = question.strip()
    
    # Check if empty
    if not question:
        return False
    
    # Check if ends with question mark
    if not question.endswith('?'):
        return False
    
    # Check if has actual content (not just punctuation and whitespace)
    content = ''.join(char for char in question if char.isalnum())
    if not content:
        return False
        
    return True

# Apply validation to the dataset
valid_questions = processed_dataset.filter(lambda x: is_valid_question(x['Question_formatted']), num_proc=32)

# Print statistics
total = len(processed_dataset)
valid = len()
print(f"\nTotal questions: {total}")
print(f"Valid questions: {valid}")
print(f"Percentage valid: {((valid)/total)*100:.2f}%")

print(f"Pushing to Hub")
valid_questions.push_to_hub("HoangHa/medical-raw", "medqa")
