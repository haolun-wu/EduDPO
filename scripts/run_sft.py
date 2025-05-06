"""
Runs Direct Preference Optimization training on our current data samples
"""

from argparse import ArgumentParser
from dotmap import DotMap
from src.models.HuggingFaceLocalModel import HuggingFaceLocalModel
from src.trl.DPO import DPO
from utils.files import load_json, load_yaml
from utils.seed import set_seed

from datasets import Dataset  

def parse_args():
    parser = ArgumentParser(description='Run DPO')
    parser.add_argument('--input_file', type=str, default="data/simulated/questions_ta_feedbacks.json",
                        help='Input JSON file containing the DPO processed data')
    parser.add_argument('--model_config', type=str, default='config/model/mistral-7b.yaml',
                        help='Path towards model configuration file')
    parser.add_argument('--training_config', type=str, default='config/task/train/train_sft.yaml',
                        help='Path towards training configuration')
    parser.add_argument('--save_dir', type=str, default='./sft_output',
                        help='Path towards the traininig saving directory')
    return parser.parse_args()


def prepare_data(file_path):
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
    
    dataset = dataset.train_test_split(test_size=0.1)

    return dataset 

def main():
    args = parse_args()
    model_config = DotMap(load_yaml(args.model_config))
    model_agent = HuggingFaceLocalModel(model_config) 

    training_config = DotMap(load_yaml(args.training_config))
    
    dataset_dict = prepare_data(args.input_file)

    set_seed(training_config.seed)
    
    dpo = DPO(model_agent, training_config, args.save_dir)
    dpo.run(dataset_dict)


if __name__ == "__main__":
    main()