import argparse
import subprocess
import os
import sys

def setup_python_path():
    """Add the project root to Python path"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return f"PYTHONPATH={project_root}"

def run_student_answers(args):
    print("\n=== Step 1: Generating student answers ===")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    subprocess.run([
        "python", "scripts/run_stu_answers.py",
        "--input_file", args.input_file,
        "--output_file", args.stu_output,
        "--model", args.stu_model,
        "--num_simulations", str(args.num_simulations),
        "--removal_probability", str(args.removal_probability),
        "--number_modification_range", str(args.number_modification_range),
        "--number_modification_chance", str(args.number_modification_chance)
    ], check=True, env=env)
    print("Student answers generation completed!")

def run_llm_feedback(args):
    print("\n=== Step 2: Generating LLM feedbacks ===")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    subprocess.run([
        "python", "scripts/run_llm_feedbacks.py",
        "--input_file", args.stu_output,
        "--output_file", args.llm_output,
        "--model_names", *args.llm_models
    ], check=True, env=env)
    print("LLM feedbacks generation completed!")

def run_ta_feedback(args):
    print("\n=== Step 3: Generating TA feedback ===")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    subprocess.run([
        "python", "scripts/run_ta_feedbacks.py",
        "--input_file", args.llm_output,
        "--output_file", args.ta_output,
        "--model", args.ta_model,
        "--max_prompt_tokens", str(args.max_prompt_tokens)
    ], check=True, env=env)
    print("TA feedback generation completed!")

def main():
    parser = argparse.ArgumentParser(description='Run the full data synthesis pipeline using subprocesses')
    
    # Input/Output paths
    parser.add_argument('--input_file', type=str, default='data/raw/questions.json')
    parser.add_argument('--stu_output', type=str, default='data/simulated/questions_stu_answers.json')
    parser.add_argument('--llm_output', type=str, default='data/simulated/questions_llm_feedbacks.json')
    parser.add_argument('--ta_output', type=str, default='data/simulated/questions_ta_feedbacks.json')
    
    # Model configurations
    parser.add_argument('--stu_model', type=str, default='mistralai/Mistral-7B-Instruct-v0.3')
    parser.add_argument('--llm_models', nargs='+', type=str, 
                        default=['allenai/OLMo-2-1124-7B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct'])
    parser.add_argument('--ta_model', type=str, default='Qwen/Qwen2.5-14B-Instruct')
    
    # Student answer generation parameters
    parser.add_argument('--num_simulations', type=int, default=10)
    parser.add_argument('--removal_probability', type=float, default=0.2)
    parser.add_argument('--number_modification_range', type=float, default=0.2)
    parser.add_argument('--number_modification_chance', type=float, default=0.2)
    
    # TA feedback parameters
    parser.add_argument('--max_prompt_tokens', type=int, default=8192)
    
    args = parser.parse_args()
    
    # Run each step
    run_student_answers(args)
    run_llm_feedback(args)
    run_ta_feedback(args)
    
    print("\n=== Pipeline completed successfully! ===")
    print(f"Final output saved to: {args.ta_output}")

if __name__ == "__main__":
    main()
