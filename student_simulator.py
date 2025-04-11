import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional
from utils import randomly_remove_information, TextModificationConfig

class StudentSimulator:
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"  # For better generation
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            # attn_implementation="flash_attention_2" if self.device == "cuda" else "eager",  # Use flash attention if available
            low_cpu_mem_usage=True  # Optimize memory usage
        )
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def _create_student_prompt(self, question_text: str) -> str:
        return f"""You are a student taking a probability course. You are trying to solve the following problem:

                {question_text}

                Please provide your solution.

                Your solution:"""

    def generate_student_solution(self, question_text: str) -> str:
        prompt = self._create_student_prompt(question_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                # do_sample=True,
                # pad_token_id=self.tokenizer.pad_token_id,
                # eos_token_id=self.tokenizer.eos_token_id,
                # use_cache=True  # Enable KV cache
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the student's solution part
        solution = response.split("Your solution:")[1].strip()
        return solution

def process_questions(
    input_file: str,
    output_file: str,
    model_name: str,
    num_simulations: int = 5,
    text_modification_config: Optional[TextModificationConfig] = None
):
    # Load all questions
    with open(input_file, 'r') as f:
        questions = json.load(f)
    
    # Initialize the student simulator
    simulator = StudentSimulator(model_name=model_name)
    
    # Initialize empty results list
    results = []
    
    # Process each question
    for question in questions:
        question_id = question['id']
        base_id = question_id * num_simulations
        
        print(f"Generating {num_simulations} solutions for question {question_id}")
        for sim_idx in range(num_simulations):
            sim_id = base_id + sim_idx
            
            try:
                # For the first simulation, keep the original question
                keep_original = (sim_idx == 0)
                
                # Generate modified question text
                modified_question = randomly_remove_information(
                    question['question_text'],
                    config=text_modification_config,
                    keep_original=keep_original
                )
                
                # Generate student solution
                stu_solution = simulator.generate_student_solution(modified_question)
                
                # Create result dictionary
                result = {
                    'id': sim_id,
                    'question_text': question['question_text'],  # Keep original question text
                    'ta_solution': question['ta_solution'],  # Keep original TA solution
                    'stu_solution': stu_solution,
                    'modified_question': modified_question,  # Store the modified question for reference
                    'is_original': keep_original  # Flag to indicate if this is the unmodified version
                }
                
                # Add to results
                results.append(result)
                
                # Save after each simulation in case of interruption
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
                    
            except Exception as e:
                print(f"Error processing simulation {sim_idx} for question {question_id}: {str(e)}")
                continue
    
    print(f"All solutions generated and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate student solutions')
    parser.add_argument('--input_file', type=str, default='data/questions.json',
                        help='Input JSON file containing questions and TA solutions')
    parser.add_argument('--output_file', type=str, default='data/questions_stu_answers.json',
                        help='Output JSON file to save student solutions')
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-Instruct-v0.3',
                        help='Model to use for generating student solutions')
    parser.add_argument('--num_simulations', type=int, default=2,
                        help='Number of student simulations to generate per question')
    parser.add_argument('--removal_probability', type=float, default=0.3,
                        help='Probability of removing a sentence')
    parser.add_argument('--number_modification_range', type=float, default=0.2,
                        help='Maximum percentage change for number modifications')
    parser.add_argument('--number_modification_chance', type=float, default=0.5,
                        help='Probability of modifying each number')
    
    args = parser.parse_args()
    
    # Create text modification configuration
    config = TextModificationConfig(
        removal_probability=args.removal_probability,
        number_modification_range=args.number_modification_range,
        number_modification_chance=args.number_modification_chance
    )
    
    process_questions(
        input_file=args.input_file,
        output_file=args.output_file,
        model_name=args.model,
        num_simulations=args.num_simulations,
        text_modification_config=config
    ) 