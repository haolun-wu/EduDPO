import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional
from utils.text_processing import randomly_remove_information, TextModificationConfig
from tqdm import tqdm

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
            attn_implementation="flash_attention_2" if self.device == "cuda" else "eager",  # Use flash attention if available
        )
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def _create_student_prompt(self, question_text: str) -> str:
        messages = [
            {"role": "system", "content": "You are a student taking a probability course."},
            {"role": "user", "content": (
                f"You are trying to solve the following problem:\n\n"
                f"{question_text}\n\n"
                "Please provide your solution.\n\n"
                "Your solution:"
            )}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False)

    def generate_student_solution(self, question_text: str) -> str:
        prompt = self._create_student_prompt(question_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                pad_token_id=self.tokenizer.pad_token_id,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Still keeping your logic to extract the solution part after "Your solution:"
        if "Your solution:" in response:
            solution = response.split("Your solution:")[-1].strip()
        else:
            solution = response.strip()
        
        return solution
        
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
    
    # Create a list of all tasks (questions × simulations)
    total_tasks = [(q, i) for q in questions for i in range(num_simulations)]
    
    # Process all tasks with a single progress bar
    for question, sim_idx in tqdm(total_tasks, desc="Generating student answers"):
        question_id = question['id']
        sim_id = question_id * num_simulations + sim_idx
        
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
    
    del simulator.model
    del simulator.tokenizer
    torch.cuda.empty_cache()
    import gc
    gc.collect()