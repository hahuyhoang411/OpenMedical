import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def analyze_dataset_metrics(dataset):
    # Initialize lists to store data
    completion_tokens = []
    finish_reasons = []
    
    # Extract data from dataset
    for metadata_list, reasons in zip(dataset['api_metadata'], dataset['finish_reasons']):
        # Get completion tokens
        for metadata in metadata_list:
            completion_tokens.append(metadata['completion_tokens'])
        
        # Get finish reasons
        for reason in reasons:
            finish_reasons.append(reason)
    
    # Convert completion tokens to numpy array
    tokens_array = np.array(completion_tokens)
    
    # Calculate token statistics
    token_stats = {
        'mean': np.mean(tokens_array),
        'median': np.median(tokens_array),
        'std': np.std(tokens_array),
        'min': np.min(tokens_array),
        'max': np.max(tokens_array),
        'total_samples': len(tokens_array)
    }
    
    # Calculate finish reasons distribution
    reason_counts = Counter(finish_reasons)
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Histogram of completion tokens
    plt.subplot(1, 3, 1)
    sns.histplot(tokens_array, bins=50)
    plt.title('Distribution of Completion Tokens')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    
    # Plot 2: Box plot of completion tokens
    plt.subplot(1, 3, 2)
    sns.boxplot(y=tokens_array)
    plt.title('Box Plot of Completion Tokens')
    plt.ylabel('Number of Tokens')
    
    # Plot 3: Bar plot of finish reasons
    plt.subplot(1, 3, 3)
    reasons_df = pd.DataFrame.from_dict(reason_counts, orient='index', columns=['count'])
    sns.barplot(data=reasons_df, x=reasons_df.index, y='count')
    plt.title('Distribution of Finish Reasons')
    plt.xlabel('Finish Reason')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print token statistics
    print("\nCompletion Tokens Statistics:")
    for key, value in token_stats.items():
        print(f"{key}: {value:.2f}")
    
    # Print token percentiles
    percentiles = [25, 50, 75, 90, 95, 99]
    print("\nToken Percentiles:")
    for p in percentiles:
        print(f"{p}th percentile: {np.percentile(tokens_array, p):.2f}")
    
    # Print finish reasons statistics
    print("\nFinish Reasons Distribution:")
    total_reasons = sum(reason_counts.values())
    for reason, count in reason_counts.items():
        percentage = (count / total_reasons) * 100
        print(f"{reason}: {count} ({percentage:.2f}%)")
    
    return token_stats, reason_counts, tokens_array

# Usage
token_stats, reason_counts, tokens = analyze_dataset_metrics(dataset)