"""
Runs Direct Preference Optimization training on our current data samples
"""

import gc
import torch
from argparse import ArgumentParser
from dotmap import DotMap
from src.models.HuggingFaceLocalModel import HuggingFaceLocalModel
from src.trl.SFT import SFT
from utils.files import load_json, load_yaml
from utils.seed import set_seed
from utils.text_processing import clean_text_formatting
from pathlib import Path
from datasets import Dataset, DatasetDict

# Load prompt template from config
PROMPT_CONFIG = load_yaml('config/task/prompt/guideline_generate_feedback.yaml')
PROMPT_TEMPLATE = PROMPT_CONFIG['prompt_template']  

"""def prepare_dataframe(df, tokenizer):
    text = [tokenizer.apply_chat_template(cleaner_eval(instr), tokenize=False) for instr in df["sft_training_completed"]]
    df["text"] = text 
    return df """


def parse_args():
    parser = ArgumentParser(description='Run DPO')
    parser.add_argument('--input_file', type=str, default="data/processed/questions_ta_feedbacks_train.json",
                        help='Input JSON file containing the TA answers')
    parser.add_argument('--model_config', type=str, default='config/model/llama3.1-8b-instruct.yaml',
                        help='Path towards model configuration file')
    parser.add_argument('--training_configs', nargs='+', type=str, 
                        default=["config/task/train/train_sft_v1.yaml"],
                        help='Paths towards training configuration files to run sequentially')
    parser.add_argument('--save_dir', type=str, default='./model_output',
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
    else:
        raise ValueError(f"Invalid data type {type(data)}")
    
    def add_chat_template(example):
        # Generate prompt directly using the template
        text = PROMPT_TEMPLATE.format(
            question_text=example["question_text"],
            ta_solution=example["ta_solution"],
            stu_solution=example["stu_solution"]
        )
        # text = clean_text_formatting(text)
        
        instruction = [{"role": "user", "content": text}, 
                      {"role": "assistant", "content": example["ta_feedback"]}]
        print("instruction is", instruction)
        example["text"] = tokenizer.apply_chat_template(instruction, 
                                                      tokenize=False)
        
        return example 
    
    dataset = dataset.map(add_chat_template, batched=False)
    print("Example SFT step", dataset[0]["text"])
    # dataset = dataset.train_test_split(test_size=0.1)
    dataset_dict = DatasetDict({
        "train": dataset,
        "test": dataset
    })


    return dataset_dict 


def clear_memory():
    """Clear GPU memory and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def main():
    
    args = parse_args()
    # Initialize model and dataset
    model_config = DotMap(load_yaml(args.model_config))
    model_agent = HuggingFaceLocalModel(model_config)
    dataset_dict = prepare_data(args.input_file, model_agent.tokenizer)
    
    # Run each training configuration sequentially
    for config_path in args.training_configs:
        print(f"\n=== Loading training configuration: {config_path} ===")
        training_config = DotMap(load_yaml(config_path))
        
        # Create save directory with training config name
        save_dir = Path(args.save_dir) / training_config.name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Training with configuration: {training_config.name}")
        print(f"Output will be saved to: {save_dir}")
        
        # Initialize trainer
        set_seed(training_config.seed)
        sft = SFT(model_agent, training_config, str(save_dir))
        sft.run(dataset_dict)
        
        # Clear memory before next configuration
        print(f"\n=== Completed training with {training_config.name} ===")
        print("Clearing memory...")
        clear_memory()
    
    print("\n=== All training configurations completed ===")


if __name__ == "__main__":
    main()