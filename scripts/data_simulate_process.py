#!/usr/bin/env python3
import argparse
import gc
import torch
from src.data.student_simulator import process_student_answers, TextModificationConfig
from src.data.llm_feedback_simulator import process_llm_feedback
from src.data.ta_feedback_simulator import process_ta_feedback
from src.data.dpo_data_converter import convert_to_dpo_samples, convert_to_dpo_parallel_format

def cleanup_memory():
    """Clean up memory by clearing CUDA cache and running garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def main():
    parser = argparse.ArgumentParser(description='Run the complete data simulation and processing pipeline')
    
    # Student simulation arguments
    parser.add_argument('--input_file', type=str, default='data/raw/questions.json',
                        help='Input JSON file containing questions and TA solutions')
    parser.add_argument('--student_output', type=str, default='data/simulated/questions_stu_answers.json',
                        help='Output JSON file to save student solutions')
    parser.add_argument('--student_model', type=str, default='mistralai/Mistral-7B-Instruct-v0.3',
                        help='Model to use for generating student solutions')
    parser.add_argument('--num_simulations', type=int, default=5,
                        help='Number of student simulations to generate per question')
    parser.add_argument('--removal_probability', type=float, default=0.3,
                        help='Probability of removing a sentence')
    parser.add_argument('--number_modification_range', type=float, default=0.2,
                        help='Maximum percentage change for number modifications')
    parser.add_argument('--number_modification_chance', type=float, default=0.5,
                        help='Probability of modifying each number')
    
    # LLM feedback arguments
    parser.add_argument('--llm_output', type=str, default='data/simulated/questions_llm_feedbacks.json',
                        help='Output JSON file to save LLM feedback')
    parser.add_argument('--llm_models', nargs='+', type=str, 
                        default=['allenai/OLMo-2-1124-7B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct'],
                        help='Models to use for generating LLM feedback')
    
    # TA feedback arguments
    parser.add_argument('--ta_output', type=str, default='data/simulated/questions_ta_feedbacks.json',
                        help='Output JSON file to save TA feedback')
    parser.add_argument('--ta_model', type=str, default='microsoft/Phi-4-mini-instruct',
                        help='Model to use for generating TA feedback')
    parser.add_argument('--max_prompt_tokens', type=int, default=8000,
                        help='Maximum number of tokens allowed in the prompt')
    
    # DPO conversion arguments
    parser.add_argument('--dpo_output', type=str, default='data/processed/dpo_samples.json',
                        help='Output JSON file to save DPO samples')
    parser.add_argument('--dpo_parallel_output', type=str, default='data/processed/dpo_parallel_samples.json',
                        help='Output JSON file to save DPO parallel samples')
    
    args = parser.parse_args()
    
    try:
        # Step 1: Generate student solutions
        print("\nStep 1: Generating student solutions...")
        config = TextModificationConfig(
            removal_probability=args.removal_probability,
            number_modification_range=args.number_modification_range,
            number_modification_chance=args.number_modification_chance
        )
        process_student_answers(
            input_file=args.input_file,
            output_file=args.student_output,
            model_name=args.student_model,
            num_simulations=args.num_simulations,
            text_modification_config=config
        )
        print("Cleaning up memory after student simulation...")
        cleanup_memory()
        
        # Step 2: Generate LLM feedback
        print("\nStep 2: Generating LLM feedback...")
        process_llm_feedback(
            input_file=args.student_output,
            output_file=args.llm_output,
            model_names=args.llm_models
        )
        print("Cleaning up memory after LLM feedback generation...")
        cleanup_memory()
        
        # Step 3: Generate TA feedback
        print("\nStep 3: Generating TA feedback...")
        process_ta_feedback(
            input_file=args.llm_output,
            output_file=args.ta_output,
            model_name=args.ta_model,
            max_prompt_tokens=args.max_prompt_tokens
        )
        print("Cleaning up memory after TA feedback generation...")
        cleanup_memory()
        
        # Step 4: Convert to DPO format
        print("\nStep 4: Converting to DPO format...")
        convert_to_dpo_samples(
            input_file=args.ta_output,
            output_file=args.dpo_output
        )
        
        # Optional: Convert to parallel DPO format
        print("\nStep 5: Converting to parallel DPO format...")
        convert_to_dpo_parallel_format(
            input_file=args.ta_output,
            output_file=args.dpo_parallel_output
        )
        
        print("\nAll steps completed successfully!")
        
    except Exception as e:
        print(f"\nError occurred during processing: {str(e)}")
        raise
    finally:
        # Final cleanup
        print("\nPerforming final memory cleanup...")
        cleanup_memory()

if __name__ == "__main__":
    main() 