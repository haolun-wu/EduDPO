import argparse
from src.data.llm_feedbacks_simulator import process_llm_feedback

def main():
    parser = argparse.ArgumentParser(description='Generate feedback using LLM')
    parser.add_argument('--input_file', type=str, default='data/simulated/questions_stu_answers.json')
    parser.add_argument('--output_file', type=str, default='data/simulated/questions_llm_feedbacks.json')
    parser.add_argument('--model_names', nargs='+', type=str, 
                        default=['allenai/OLMo-2-1124-13B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct'])
    
    # 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
    # 'deepseek-ai/DeepSeek-V2-Lite-Chat'
    # 'deepseek-ai/deepseek-moe-16b-chat'
    args = parser.parse_args()
    
    # Generate LLM feedback
    print("Generating LLM feedback...")
    process_llm_feedback(
        input_file=args.input_file,
        output_file=args.output_file,
        model_names=args.model_names
    )
    
    print("\nLLM feedback generation completed!")

if __name__ == "__main__":
    main() 