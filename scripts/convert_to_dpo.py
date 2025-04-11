import argparse
from src.data.dpo_data_converter import convert_to_dpo_samples, convert_to_dpo_parallel_format

def main():
    parser = argparse.ArgumentParser(description='Convert TA feedback data to DPO training samples')
    parser.add_argument('--input_file', type=str, default='data/simulated/questions_ta_feedbacks.json',
                        help='Input JSON file containing TA feedback data')
    parser.add_argument('--output_file1', type=str, default='data/processed/dpo_training_samples.json',
                        help='Output JSON file for individual DPO samples')
    parser.add_argument('--output_file2', type=str, default='data/processed/dpo_training_samples_parallel.json',
                        help='Output JSON file for parallel format DPO samples')
    
    args = parser.parse_args()
    
    # Convert to individual samples format
    convert_to_dpo_samples(args.input_file, args.output_file1)
    
    # Convert to parallel format
    convert_to_dpo_parallel_format(args.input_file, args.output_file2)

if __name__ == "__main__":
    main() 