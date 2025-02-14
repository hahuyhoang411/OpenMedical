import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_model_accuracy(dataset):
    # Initialize lists to store accuracies for each attempt count
    accuracy_1 = []
    accuracy_2 = []
    accuracy_3 = []
    
    # Process each row in the dataset
    for correctness_list in dataset['correctness']:
        # Get accuracy for different number of attempts
        accuracy_1.append(correctness_list[0])  # First attempt
        accuracy_2.append(any(correctness_list[:2]))  # First two attempts
        accuracy_3.append(any(correctness_list))  # All three attempts
    
    # Calculate accuracy percentages
    acc_1 = np.mean(accuracy_1) * 100
    acc_2 = np.mean(accuracy_2) * 100
    acc_3 = np.mean(accuracy_3) * 100
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Pie chart for 1 attempt
    ax1.pie([acc_1, 100-acc_1], labels=['Correct', 'Incorrect'], 
            autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    ax1.set_title('Accuracy with 1 Attempt')
    
    # Plot 2: Pie chart for 2 attempts
    ax2.pie([acc_2, 100-acc_2], labels=['Correct', 'Incorrect'], 
            autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    ax2.set_title('Accuracy with 2 Attempts')
    
    # Plot 3: Pie chart for 3 attempts
    ax3.pie([acc_3, 100-acc_3], labels=['Correct', 'Incorrect'], 
            autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    ax3.set_title('Accuracy with 3 Attempts')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print("\nModel Accuracy Statistics:")
    print(f"Accuracy with 1 attempt: {acc_1:.2f}%")
    print(f"Accuracy with 2 attempts: {acc_2:.2f}%")
    print(f"Accuracy with 3 attempts: {acc_3:.2f}%")
    
    # Calculate improvement statistics
    print("\nImprovement Statistics:")
    print(f"Improvement from 1 to 2 attempts: {(acc_2 - acc_1):.2f}%")
    print(f"Improvement from 2 to 3 attempts: {(acc_3 - acc_2):.2f}%")
    print(f"Total improvement (1 to 3 attempts): {(acc_3 - acc_1):.2f}%")
    
    return {
        'accuracy_1': acc_1,
        'accuracy_2': acc_2,
        'accuracy_3': acc_3,
    }

# Usage
accuracy_stats = analyze_model_accuracy(dataset)