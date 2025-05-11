""" Training script for running inference of a trained model over
a validation or testing dataset."""

import os
import gc
import re
from src.judge.judging import generate_judging_prompt
import torch

import json
from argparse import ArgumentParser
from dotmap import DotMap
from src.data.dpo_data_converter import generate_DPO_training_prompt
from src.models.HuggingFaceLocalModel import HuggingFaceLocalModel
from src.models.Inference import Inference
from utils.distance import rougelcsum_dist
from utils.files import load_json, load_yaml

from datasets import Dataset  


def parse_args():
    parser = ArgumentParser(description='Run Inference')
    parser.add_argument('--input_file', type=str, default="data/inference/llama3.1-8b-instruct_questions_ta_feedbacks_train.json",
                        help='Input JSON file containing TA answers')
    parser.add_argument('--model_config', type=str, default='config/model/llama3.1-8b-instruct.yaml',
                        help='Path towards model configuration file that is going to judge')
    parser.add_argument('--save_dir', type=str, default='./data/judge/',
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
        text = generate_judging_prompt(example["question_text"], 
                                       example["ta_solution"], 
                                       example["stu_solution"],
                                       example["feedback"])
        instruction = [{"role": "user", "content": text}]
        text = tokenizer.apply_chat_template(instruction, 
                                             add_generation_prompt=True, 
                                             tokenize=False)

        example["text"] = text

        return example 
    
    dataset = dataset.map(add_chat_template, batched=False)
    print("Example judging step", dataset[0]["text"])

    return dataset 


def clear_memory():
    """Clear GPU memory and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def decode_json_response(response):
    try:
        r = re.search(r'\{\s*[^{}]*?\s*\}', response).group(0)
        answer_dict = json.loads(r)
        answer_dict["response"] = response
    except Exception:
        answer_dict = {"correctness": "", "helpfulness": "", 
                       "response": response}
    
    return answer_dict
    
def main():
    
    args = parse_args()
    model_config = DotMap(load_yaml(args.model_config))
    model_agent = HuggingFaceLocalModel(model_config)
    inference_agent = Inference(model_agent.model, model_agent.tokenizer)
    dataset = prepare_data(args.input_file, model_agent.tokenizer)
    
    def generate_judging(examples):
        responses = inference_agent.pipe(examples["text"], 
                                         return_full_text=False,
                                         **{"max_new_tokens": 1024})
        responses = [resp[j]['generated_text'] 
                     for resp in responses 
                     for j in range(len(resp))]
        responses = [decode_json_response(r) for r in responses]
        
        # Create result dictionaries in the same format as stu_answers_simulator.py
        results = []
        for i, (response, example) in enumerate(zip(responses, examples)):
            result = {
                'id': example['id'],
                'question_text': example['question_text'],
                'ta_solution': example['ta_solution'],
                'stu_solution': example['stu_solution'],
                'feedback': example['feedback'],
                'correctness': response['correctness'],
                'helpfulness': response['helpfulness'],
                'response': response['response'],
                'rouge': rougelcsum_dist(example['ta_solution'], example['feedback'], get_score=True)
            }
            results.append(result)
        
        # examples["correctness"] = [r["correctness"] for r in responses]
        # examples["helpfulness"] = [r["helpfulness"] for r in responses] 
        # examples["response"] = [r["response"] for r in responses]
        # examples["rouge"] = [rougelcsum_dist(ta_sol, feedback, get_score=True) for ta_sol, feedback in zip(examples["ta_solution"], examples["feedback"])] #examples[responses]
        # return examples
        
        return results

    # Process all samples
    results = []
    for batch in dataset.iter(batch_size=2):
        batch_results = generate_judging(batch)
        results.extend(batch_results)
    
    # Extract model name from input file path
    input_model_name = args.input_file.split('/')[-1].split('_')[0]  # Get the model name from input file
    judge_model_name = args.model_config.split("/")[-1].replace(".yaml", "")
    
    # Create filename with both model names
    filename = f"{judge_model_name}_judge_{input_model_name}_{args.input_file.split('/')[-1]}"
    
    # Save results in JSON format
    with open(os.path.join(args.save_dir, filename), 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    # dataset = dataset.select(list(range(1)))
    # dataset = dataset.map(generate_judging, batched=True, batch_size=2)
    # print("dataset response", dataset["correctness"][0], dataset["response"][0], dataset["rouge"][0])
    # dataset.to_json(os.path.join(args.save_dir, filename))

if __name__ == "__main__":
    main()