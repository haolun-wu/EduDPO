import argparse
import json
import random
from pathlib import Path
from src.data.dpo_data_converter import (
    split_data,
    add_sequential_ids,
    save_json,
    convert_to_dpo_samples
)

def main():
    parser = argparse.ArgumentParser(description='Prepare data for DPO training')
    parser.add_argument('--input_file', type=str, default='data/simulated/questions_ta_feedbacks.json',
                        help='Input JSON file containing the data')
    parser.add_argument('--train_file', type=str, default='data/processed/questions_ta_feedbacks_train.json',
                        help='Output file for training set')
    parser.add_argument('--val_file', type=str, default='data/processed/questions_ta_feedbacks_val.json',
                        help='Output file for validation set')
    parser.add_argument('--test_file', type=str, default='data/processed/questions_ta_feedbacks_test.json',
                        help='Output file for test set')
    parser.add_argument('--dpo_output', type=str, default='data/processed/dpo_training_samples.json',
                        help='Output file for DPO training samples')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Ratio of data for validation')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Ratio of data for testing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Load the data
    print("Loading data...")
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    # Split the data
    print("Splitting data into train/val/test sets...")
    train_data, val_data, test_data = split_data(
        data=data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Add sequential IDs to each split (each starting from 0)
    train_data = add_sequential_ids(train_data)
    val_data = add_sequential_ids(val_data)
    test_data = add_sequential_ids(test_data)
    
    # Save the splits
    print("Saving data splits...")
    save_json(train_data, args.train_file)
    save_json(val_data, args.val_file)
    save_json(test_data, args.test_file)
    
    # Print detailed statistics
    print("\nData split statistics:")
    print(f"Total samples: {len(data)}")
    print(f"Training set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    
    # Print sample IDs for each split
    print("\nSample IDs in each split:")
    print(f"Training set IDs: {[item['id'] for item in train_data]} (original: {[item['original_id'] for item in train_data]})")
    print(f"Validation set IDs: {[item['id'] for item in val_data]} (original: {[item['original_id'] for item in val_data]})")
    print(f"Test set IDs: {[item['id'] for item in test_data]} (original: {[item['original_id'] for item in test_data]})")
    
    # Convert training set to DPO format
    print("\nConverting training set to DPO format...")
    convert_to_dpo_samples(args.train_file, args.dpo_output)
    print(f"DPO training samples generated and saved to {args.dpo_output}")

if __name__ == "__main__":
    main() 