"""
Runs Direct Preference Optimization training on our current data samples
"""

import gc
import torch

from argparse import ArgumentParser
from dotmap import DotMap
from src.models.HuggingFaceLocalModel import HuggingFaceLocalModel
from src.trl.DPO import DPO
from utils.files import load_json, load_yaml
from utils.seed import set_seed
from pathlib import Path
from datasets import Dataset  

def parse_args():
    parser = ArgumentParser(description='Run DPO')
    parser.add_argument('--input_file', type=str, default="data/processed/dpo_training_samples.json",
                        help='Input JSON file containing the DPO processed data')
    parser.add_argument('--model_config', type=str, default='config/model/llama3.1-8b-instruct.yaml',
                        help='Path towards model configuration file')
    parser.add_argument('--training_config', type=str, 
                        default='config/task/train/train_rpo.yaml',
                        help='Path towards training configuration file')
    parser.add_argument('--save_dir', type=str, default='./dpo_output',
                        help='Base path for saving outputs')
    return parser.parse_args()





def prepare_data(file_path, tokenizer):
    """ 
    Loads data from a file and
    return a HuggingFace DatasetDict object 
    with training and testing splits
    """

    data = load_json(file_path)
    if type(data) == list:
        dataset = Dataset.from_list(data)
    elif type(data) == dict:
        dataset = Dataset.from_dict(data)
    else:
        raise ValueError(f"Invalid data type {type(data)}")
    
    def add_chat_template(example):
        original_prompt = example["prompt"]
        instruction = [{"role": "user", "content": original_prompt}]
        example["prompt"] = tokenizer.apply_chat_template(instruction, 
                                                          add_generation_prompt=True, 
                                                          tokenize=False)

        instruction.append({"role": "assistant", "content": example["chosen"]})
        example["chosen"] = tokenizer.apply_chat_template(instruction, tokenize=False)
        
        instruction = [{"role": "user", "content": original_prompt}]
        instruction.append({"role": "assistant", "content": example["rejected"]})
        example["rejected"] = tokenizer.apply_chat_template(instruction, tokenize=False)

        return example 
    
    dataset = dataset.map(add_chat_template, batched=False)
    dataset = dataset.train_test_split(test_size=0.1)

    return dataset 

def clear_memory():
    """Clear GPU memory and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def main():
    
    args = parse_args()
    # Initialize model and dataset
    model_config = DotMap(load_yaml(args.model_config))
    training_config = DotMap(load_yaml(args.training_config))
    
    # Create save directory with training config name
    save_dir = Path(args.save_dir) / training_config.name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training with configuration: {training_config.name}")
    print(f"Output will be saved to: {save_dir}")
    
    # Initialize model and trainer
    set_seed(training_config.seed)
    model_agent = HuggingFaceLocalModel(model_config)
    dataset_dict = prepare_data(args.input_file, model_agent.tokenizer)
    dpo = DPO(model_agent, training_config, str(save_dir))
    dpo.run(dataset_dict)
    
    print(f"\n=== Completed training with {training_config.name} ===")
    # print("Clearing memory...")
    clear_memory()


if __name__ == "__main__":
    main()