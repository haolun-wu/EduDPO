"""
Using DSPY to judge the quality of provided feedback
"""

import os 
import re 
import json 
import dspy
import pandas as pd 

from argparse import ArgumentParser
from src.dspy.JudgingFeedbackSignature import JudgingFeedbackSignature, build_dspy_dataset
from utils.distance import rougelcsum_dist
from utils.files import load_json, load_yaml
from datasets import Dataset 
from dotmap import DotMap
from tqdm import tqdm 

def parse_args():
    parser = ArgumentParser(description='Run Inference')
    parser.add_argument('--input_file', type=str, default="data/inference/llama3.1-8b-instruct_questions_ta_feedbacks_train.json",
                        help='Input JSON file containing TA answers')
    parser.add_argument('--model_config', type=str, default="config/model/gpt-4.1-mini.yaml",
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

    return dataset 

def decode_json_response(response):
    print("input json to decode", response)
    try:
        r = re.search(r'\{\s*[^{}]*?\s*\}', response).group(0)
        return json.loads(r)
    except Exception:
        print("exception")
        return {}
    
def main():
    args = parse_args()
    
    # Getting the dataframe 
    data = load_json(args.input_file)
    df = pd.DataFrame(data)
    dspy_dataset = build_dspy_dataset(df)

    # Configuring the model used for the predictions
    model_config = DotMap(load_yaml(args.model_config))
    lm = dspy.LM(f'{model_config.source}/{model_config.name}', 
                    api_key=os.environ["OPENAI_API_KEY"], 
                    temperature=0.0, top_p=1.0, max_tokens=5000, stop=None)
    dspy.configure(lm=lm)

    # Configuring the prompt for that generation 
    judging_task = dspy.ChainOfThought(JudgingFeedbackSignature)

    output_dataframe = []
    for i, x in enumerate(tqdm(dspy_dataset)):
        pred = judging_task(**x.inputs())
        output_dataframe.append({
            "reasoning": pred.reasoning,
            "assessment": pred.asessment,
            "rouge": rougelcsum_dist(df.iloc[i].ta_feedback, 
                                     x.feedback, get_score=True),
            "messages": lm.history[-1]["messages"],
            "cost": lm.history[-1]["cost"],
            "outputs": lm.history[-1]["outputs"],
            **pred.asessment,
        })

        print("----TA original feedback----")
        print(df.iloc[i].ta_feedback)
        print("----Generated feedback after training----")
        print(x.feedback)
        print("-----")
        print()
        print()

    dataset = Dataset.from_list(output_dataframe)
    filename = args.model_config.split("/")[-1].replace(".yaml", "")
    filename = filename + "_" + args.input_file.split("/")[-1]#.replace(".json", "")
    dataset.to_json(os.path.join(args.save_dir, filename))
    print("Total cost of evaluation", sum(dataset["cost"]))

if __name__ == "__main__":
    main()