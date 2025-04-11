import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional

class FeedbackGenerator:
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize LLM model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        
        # Set pad token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _create_prompt(self, question: str, ta_solution: str, stu_solution: str) -> str:
        return f"""Please act like a teaching assistant in a probability course. 
                    Your task is to provide detailed feedback on a student's solution to a probability problem.
                    You should first state whether the student's solution is correct or not.

                Problem:
                {question}

                Suggested Solution:
                {ta_solution}

                Student's Solution:
                {stu_solution}
                
                Your feedback:"""

    def generate_feedback(self, question: str, ta_solution: str, stu_solution: str) -> str:
        # Generate LLM feedback
        prompt = self._create_prompt(question, ta_solution, stu_solution)
        # print("\nPrompt sent to LLM:")
        # print(prompt)
        # print("\n")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                # temperature=0.7,
                # top_p=0.9
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print("\nFull LLM Response:")
        # print(full_response[:20])
        # print("\n")
        
        # Extract feedback after "Your feedback:"
        if "Your feedback:" in full_response:
            feedback = full_response.split("Your feedback:")[1].strip()
        else:
            feedback = full_response.strip()
        
        return feedback

def process_feedback(input_file: str, output_file: str, model_names: List[str]):
    # Load the questions and solutions
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    if len(model_names) != 2:
        raise ValueError("Exactly two model names must be provided: first for base_feedback, second for llama_feedback")
    
    # Initialize feedback generators for each model
    base_generator = FeedbackGenerator(model_name=model_names[0])
    llama_generator = FeedbackGenerator(model_name=model_names[1])
    
    # Process each question
    results = []
    for item in data:
        try:
            # Get feedback from both models
            base_feedback = base_generator.generate_feedback(
                question=item['question_text'],
                ta_solution=item['ta_solution'],
                stu_solution=item['stu_solution']
            )
            # print("base_feedback:", base_feedback[:10])
            
            llama_feedback = llama_generator.generate_feedback(
                question=item['question_text'],
                ta_solution=item['ta_solution'],
                stu_solution=item['stu_solution']
            )
            # print("llama_feedback:", llama_feedback[:10])
            
            # Create result dictionary with both feedbacks
            results.append({
                'id': item['id'],
                'question_text': item['question_text'],
                'ta_solution': item['ta_solution'],
                'stu_solution': item['stu_solution'],
                'base_feedback': base_feedback,
                'llama_feedback': llama_feedback
            })
            
        except Exception as e:
            print(f"Error processing question {item['id']}: {str(e)}")
            continue
    
    # Save the results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate feedback using LLM')
    parser.add_argument('--input_file', type=str, default='data/questions_stu_answers.json')
    parser.add_argument('--output_file', type=str, default='data/questions_llm_feedbacks.json')
    parser.add_argument('--model_names', nargs='+', type=str, 
                        default=['allenai/OLMo-2-1124-7B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct'])
    
    # 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
    # 'deepseek-ai/DeepSeek-V2-Lite-Chat'
    # 'deepseek-ai/deepseek-moe-16b-chat'
    args = parser.parse_args()
    process_feedback(
        input_file=args.input_file,
        output_file=args.output_file,
        model_names=args.model_names
    )