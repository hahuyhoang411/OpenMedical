import argparse
import hashlib
import json
from collections import defaultdict
from datasets import load_dataset
from pathlib import Path
from typing import Dict, List, Set, Tuple
import uuid  # Import the uuid library


def generate_uuid(text: str) -> str:
    """Generate a consistent UUID from text using MD5."""
    return hashlib.md5(str(text).encode()).hexdigest()


def analyze_duplicates(dataset, uuid_column: str, text_column: str) -> Tuple[Dict, Dict, Set]:
    """
    Analyze duplicates in the dataset based on UUIDs and content.

    Returns:
        - uuid_to_indices: Mapping of UUIDs to their indices in dataset
        - content_to_indices: Mapping of content hashes to their indices
        - duplicate_indices: Set of indices that are duplicates
    """
    uuid_to_indices = defaultdict(list)
    content_to_indices = defaultdict(list)
    duplicate_indices = set()

    for idx, example in enumerate(dataset):
        # Generate UUID from the specified column
        uuid_val = generate_uuid(example[uuid_column])
        uuid_to_indices[uuid_val].append(idx)

        # Also check content duplicates
        content = example[text_column]
        content_hash = generate_uuid(content)
        content_to_indices[content_hash].append(idx)

    # Find duplicates
    for uuid_val, indices in uuid_to_indices.items():
        if len(indices) > 1:
            # Keep the first occurrence, mark others as duplicates
            duplicate_indices.update(indices[1:])

    for content_hash, indices in content_to_indices.items():
        if len(indices) > 1:
            # Keep the first occurrence, mark others as duplicates
            duplicate_indices.update(indices[1:])

    return uuid_to_indices, content_to_indices, duplicate_indices


def main():
    parser = argparse.ArgumentParser(description='Analyze and deduplicate dataset based on UUIDs')
    parser.add_argument('--dataset-name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--uuid-column', type=str, required=True, help='Column to use for UUID generation')
    parser.add_argument('--text-column', type=str, required=True, help='Column containing the main text content')
    parser.add_argument('--output-dir', type=str, default='dedup_output', help='Output directory')
    parser.add_argument('--save-cleaned', action='store_true', help='Save the deduplicated dataset')
    parser.add_argument('--continue-generation', action='store_true',
                        help='Continue to add uuid for existed dataset')  # New argument

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load dataset
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")
    original_size = len(dataset)
    print(f"Original dataset size: {original_size}")

    if args.continue_generation:
        if "uuid" in dataset.column_names:
            raise ValueError("The dataset already contains a 'uuid' column. Cannot continue generation.")

        print("Adding 'uuid' column...")
        dataset = dataset.map(lambda example: {"uuid": str(uuid.uuid4())})  # Use standard uuid4
        original_size = len(dataset)

    else:
        # Analyze duplicates
        print("Analyzing duplicates...")
        uuid_to_indices, content_to_indices, duplicate_indices = analyze_duplicates(
            dataset, args.uuid_column, args.text_column
        )

        # Generate statistics
        uuid_duplicates = sum(1 for indices in uuid_to_indices.values() if len(indices) > 1)
        content_duplicates = sum(1 for indices in content_to_indices.values() if len(indices) > 1)

        # Save analysis results
        analysis = {
            "original_size": original_size,
            "unique_uuids": len(uuid_to_indices),
            "unique_contents": len(content_to_indices),
            "uuid_duplicates": uuid_duplicates,
            "content_duplicates": content_duplicates,
            "total_duplicates": len(duplicate_indices),
            "final_size": original_size - len(duplicate_indices)
        }

        # Print and save analysis
        print("\nAnalysis Results:")
        print(f"Original dataset size: {analysis['original_size']}")
        print(f"Number of unique UUIDs: {analysis['unique_uuids']}")
        print(f"Number of unique contents: {analysis['unique_contents']}")
        print(f"UUID duplicates found: {analysis['uuid_duplicates']}")
        print(f"Content duplicates found: {analysis['content_duplicates']}")
        print(f"Total duplicates to remove: {analysis['total_duplicates']}")
        print(f"Final dataset size after deduplication: {analysis['final_size']}")

        # Save analysis to file
        with open(output_dir / 'analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)

        # Save duplicate indices
        with open(output_dir / 'duplicate_indices.json', 'w') as f:
            json.dump(list(duplicate_indices), f)

        # Save mapping of UUIDs to help with debugging
        uuid_mapping = {
            idx: generate_uuid(example[args.uuid_column])
            for idx, example in enumerate(dataset)
        }
        with open(output_dir / 'uuid_mapping.json', 'w') as f:
            json.dump(uuid_mapping, f, indent=2)
        keep_indices = set(range(len(dataset))) - duplicate_indices
        cleaned_dataset = dataset.select(list(keep_indices))

        # Save deduplicated dataset if requested
        if args.save_cleaned:
            print("\nSaving deduplicated dataset...")
            dataset = cleaned_dataset

    if args.save_cleaned or args.continue_generation:  # Save if saving cleaned or continuing generation.
        print("\nSaving modified dataset...")
        if args.continue_generation:
          dataset.save_to_disk(output_dir / 'generated_dataset')
        else:
          dataset.save_to_disk(output_dir / 'cleaned_dataset')
        print(f"Saved modified dataset with {len(dataset)} examples")

    print(f"\nAll analysis files saved to: {output_dir}")


if __name__ == "__main__":
    main()

# Usage
# python dedup.py     --dataset-name OpenMedical/m1-raw-qwen7b-distil     --uuid-column question     --text-column question     --output-dir dedup_qwen7b_distil     --save-cleaned