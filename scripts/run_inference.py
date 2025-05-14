""" Training script for running inference of a trained model over
a validation or testing dataset."""

import os
import gc
import re
import torch
import json
from argparse import ArgumentParser
from dotmap import DotMap
from utils.files import load_json, load_yaml
from src.models.HuggingFaceLocalModel import HuggingFaceLocalModel
from src.models.Inference import Inference
from utils.files import load_json, load_yaml

from datasets import Dataset  

# Load prompt template from config
PROMPT_CONFIG = load_yaml('config/task/prompt/guideline_generate_feedback.yaml')
PROMPT_TEMPLATE = PROMPT_CONFIG['prompt_template']


def parse_args():
    parser = ArgumentParser(description='Run Inference')
    parser.add_argument('--input_file', type=str, default="data/processed/questions_ta_feedbacks_train.json",
                        help='Input JSON file containing TA answers')
    parser.add_argument('--model_config', type=str, default='config/model/llama3.1-8b-instruct.yaml',
                        help='Path towards model configuration file')
    parser.add_argument('--train_folder', type=str, default="./model_output",
                        help='Path towards a training configuration file')
    parser.add_argument('--save_dir', type=str, default='./data/inference/',
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
        text = PROMPT_TEMPLATE.format(
            question_text=example["question_text"],
            ta_solution=example["ta_solution"],
            stu_solution=example["stu_solution"]
        )
        instruction = [{"role": "user", "content": text}]
        text = tokenizer.apply_chat_template(instruction, 
                                             add_generation_prompt=True, 
                                             tokenize=False)

        example["text"] = text

        return example 
    
    dataset = dataset.map(add_chat_template, batched=False)
    print("Example SFT step", dataset[0]["text"])

    return dataset 


def clear_memory():
    """Clear GPU memory and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    
def main():
    
    args = parse_args()
    model_config = DotMap(load_yaml(args.model_config))
    is_adapter = False
    if args.train_folder is not None:
        is_adapter = True
        model_config.name = args.train_folder
    
    model_agent = HuggingFaceLocalModel(model_config, is_adapter)
    inference_agent = Inference(model_agent.model, model_agent.tokenizer)
    dataset = prepare_data(args.input_file, model_agent.tokenizer)
    
    def generate_feedback(examples):
        responses = inference_agent.pipe(examples["text"], 
                                         return_full_text=False,
                                         **{"max_new_tokens": 1024})
        responses = [resp[j]['generated_text'] 
                     for resp in responses 
                     for j in range(len(resp))]
        print("responses", responses)
        
        examples["infer_feedback"] = responses
        return examples

    # dataset = dataset.select(list(range(10)))
    dataset = dataset.map(generate_feedback, batched=True, batch_size=2)
    # print("dataset feedback and response", dataset["feedback"][0], dataset["response"][0])
    
    # Extract model name from train_folder path
    if args.train_folder:
        model_name = args.train_folder.split('/')[-1]  # Get the last part of the path
    else:
        model_name = args.model_config.split("/")[-1].replace(".yaml", "")
    filename = f"{model_name}_{args.input_file.split('/')[-1]}"
    
    # Save with indentation for better readability, matching the example structure
    data = []
    for i in range(len(dataset)):
        example = {col: dataset[col][i] for col in dataset.column_names}
        data.append(example)
    
    with open(os.path.join(args.save_dir, filename), 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()