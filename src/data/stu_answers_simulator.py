import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional
from utils.text_processing import randomly_remove_information, TextModificationConfig

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
        messages = [
            {"role": "system", "content": "You are a student taking a probability course."},
            {"role": "user", "content": f"You are trying to solve the following problem:\n\n{question_text}\n\nPlease provide your solution.\n\nYour solution:"}
        ]
        # Use chat template to format the messages into a prompt string
        return self.tokenizer.apply_chat_template(messages, tokenize=False)


    def generate_student_solution(self, question_text: str) -> str:
        prompt = self._create_student_prompt(question_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Still keeping your logic to extract the solution part after "Your solution:"
        if "Your solution:" in response:
            solution = response.split("Your solution:")[-1].strip()
        else:
            solution = response.strip()
        
        return solution
        
    # def _create_student_prompt(self, question_text: str) -> str:
    #     return f"""You are a student taking a probability course. You are trying to solve the following problem:

    #             {question_text}

    #             Please provide your solution.

    #             Your solution:"""

    # def generate_student_solution(self, question_text: str) -> str:
    #     prompt = self._create_student_prompt(question_text)
    #     inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
    #     with torch.no_grad():
    #         outputs = self.model.generate(
    #             **inputs,
    #             max_new_tokens=1024,
    #             temperature=0.7,
    #             top_p=0.9,
    #             # do_sample=True,
    #             # pad_token_id=self.tokenizer.pad_token_id,
    #             # eos_token_id=self.tokenizer.eos_token_id,
    #             # use_cache=True  # Enable KV cache
    #         )
        
    #     response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     # Extract only the student's solution part
    #     solution = response.split("Your solution:")[1].strip()
    #     return solution

def process_student_answers(
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