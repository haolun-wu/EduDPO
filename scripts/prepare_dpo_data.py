import argparse
import json
import random
from pathlib import Path
from src.data.dpo_data_converter import convert_to_dpo_samples

def split_data(data: list, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1,
               seed: int = 42):
    """
    Split the data into train, validation, and test sets.
    
    Args:
        data: List of data samples
        train_ratio: Ratio of data for training (default: 0.8)
        val_ratio: Ratio of data for validation (default: 0.1)
        test_ratio: Ratio of data for testing (default: 0.1)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Set random seed
    random.seed(seed)
    
    # Shuffle the data
    random.shuffle(data)
    
    # Calculate split indices
    n = len(data)
    # Calculate exact number of samples for each split
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val  # Use remaining samples for test set
    
    # Split the data
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]
    
    return train_data, val_data, test_data

def add_sequential_ids(data: list):
    """
    Add sequential IDs to the data while preserving original IDs.
    Each split will have IDs starting from 0.
    
    Args:
        data: List of data samples
        
    Returns:
        List of data samples with new sequential IDs
    """
    for i, item in enumerate(data):
        item['original_id'] = item['id']  # Preserve original ID
        item['id'] = i  # Add new sequential ID starting from 0
    return data

def save_json(data: list, file_path: str):
    """Save data to a JSON file."""
    # Create output directory if it doesn't exist
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

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