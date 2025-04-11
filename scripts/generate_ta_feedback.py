import argparse
from src.data.ta_feedback_simulator import process_ta_feedback

def main():
    parser = argparse.ArgumentParser(description='Generate TA feedback')
    parser.add_argument('--input_file', type=str, default='data/simulated/questions_llm_feedbacks.json',
                        help='Input JSON file containing questions and feedbacks')
    parser.add_argument('--output_file', type=str, default='data/simulated/questions_ta_feedbacks.json',
                        help='Output JSON file to save TA feedback')
    parser.add_argument('--model', type=str, default='microsoft/Phi-4-mini-instruct',
                        help='Model to use for generating TA feedback')
    parser.add_argument('--max_prompt_tokens', type=int, default=6000,
                        help='Maximum number of tokens allowed in the prompt')
    
    args = parser.parse_args()

    # Generate TA feedback
    print("Generating TA feedback...")
    process_ta_feedback(
        input_file=args.input_file,
        output_file=args.output_file,
        model_name=args.model,
        max_prompt_tokens=args.max_prompt_tokens
    ) 
    
    print("\nTA feedback generation completed!")

if __name__ == "__main__":
    main() 