import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from utils.text_processing import clean_text_formatting
from utils.files import load_yaml

# Load prompt template from config
PROMPT_CONFIG = load_yaml('config/task/prompt/guideline_generate_feedback.yaml')
PROMPT_TEMPLATE = PROMPT_CONFIG['prompt_template']

def convert_to_dpo_samples(input_file: str, output_file: str) -> None:
    # Load the original data
    with open(input_file, 'r') as f:
        original_data = json.load(f)
    
    dpo_samples = []
    sample_id = 0
    
    for item in original_data:
        # Generate the prompt directly using the template
        prompt = PROMPT_TEMPLATE.format(
            question_text=item['question_text'],
            ta_solution=item['ta_solution'],
            stu_solution=item['stu_solution']
        )
        
        # Create two samples for each original item
        # Sample 1: chosen = ta_feedback, rejected = base_feedback
        dpo_samples.append({
            'sample_id': sample_id,
            'id': item['id'],
            'prompt': clean_text_formatting(prompt),
            'chosen': clean_text_formatting(item['ta_feedback']),
            'rejected': clean_text_formatting(item['base_feedback']),
            'llm': 'base'
        })
        sample_id += 1
        
        # Sample 2: chosen = ta_feedback, rejected = llama_feedback
        dpo_samples.append({
            'sample_id': sample_id,
            'id': item['id'],
            'prompt': clean_text_formatting(prompt),
            'chosen': clean_text_formatting(item['ta_feedback']),
            'rejected': clean_text_formatting(item['llama_feedback']),
            'llm': 'llama'
        })
        sample_id += 1
    
    # Save the DPO samples
    with open(output_file, 'w') as f:
        json.dump(dpo_samples, f, indent=4, ensure_ascii=False)

def split_data(data: list, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1,
               seed: int = 42) -> Tuple[list, list, list]:
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

def add_sequential_ids(data: list) -> list:
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

def save_json(data: list, file_path: str) -> None:
    """Save data to a JSON file."""
    # Create output directory if it doesn't exist
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
