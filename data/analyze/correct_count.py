import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_correct_attempts(dataset):
    # Initialize counters
    correct_1_of_3 = 0  # Exactly 1 correct out of 3
    correct_2_of_3 = 0  # Exactly 2 correct out of 3
    correct_3_of_3 = 0  # All 3 correct
    total_questions = len(dataset['correctness'])
    
    # Analyze each question's correctness
    for correctness_list in dataset['correctness']:
        correct_count = sum(correctness_list)
        if correct_count == 1:
            correct_1_of_3 += 1
        elif correct_count == 2:
            correct_2_of_3 += 1
        elif correct_count == 3:
            correct_3_of_3 += 1
    
    # Calculate percentages
    percent_1_of_3 = (correct_1_of_3 / total_questions) * 100
    percent_2_of_3 = (correct_2_of_3 / total_questions) * 100
    percent_3_of_3 = (correct_3_of_3 / total_questions) * 100
    
    # Create data for plotting
    categories = ['1/3 correct', '2/3 correct', '3/3 correct']
    counts = [correct_1_of_3, correct_2_of_3, correct_3_of_3]
    percentages = [percent_1_of_3, percent_2_of_3, percent_3_of_3]
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, counts)
    
    # Add value labels on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f}\n({percentages[i]:.1f}%)',
                ha='center', va='bottom')
    
    plt.title('Distribution of Correct Answers across 3 Attempts')
    plt.ylabel('Number of Questions')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.show()
    
    # Print detailed statistics
    print("\nDetailed Statistics:")
    print(f"Total questions analyzed: {total_questions:,}")
    print(f"\nQuestions with exactly 1/3 correct: {correct_1_of_3:,} ({percent_1_of_3:.1f}%)")
    print(f"Questions with exactly 2/3 correct: {correct_2_of_3:,} ({percent_2_of_3:.1f}%)")
    print(f"Questions with all 3/3 correct: {correct_3_of_3:,} ({percent_3_of_3:.1f}%)")
    
    return {
        'exact_1_correct': correct_1_of_3,
        'exact_2_correct': correct_2_of_3,
        'exact_3_correct': correct_3_of_3,
        'total_questions': total_questions
    }

# Usage
results = analyze_correct_attempts(dataset)