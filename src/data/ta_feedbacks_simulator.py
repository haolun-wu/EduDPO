import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional
from utils.text_processing import clean_text_formatting
from utils.files import load_yaml
from tqdm import tqdm

# Load prompt template from config
PROMPT_CONFIG = load_yaml('config/task/prompt/guideline_generate_feedback.yaml')
PROMPT_TEMPLATE = PROMPT_CONFIG['prompt_template']

class TAFeedbackSimulator:
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            attn_implementation="flash_attention_2" if self.device == "cuda" else "eager",  # Use flash attention if available
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _create_ta_prompt(self, question: str, ta_solution: str, stu_solution: str, 
                        base_feedback: str, llama_feedback: str) -> list:
        # Get the base prompt with the main variables
        base_prompt = PROMPT_TEMPLATE.format(
            question_text=question,
            ta_solution=ta_solution,
            stu_solution=stu_solution
        )

        # Add the LLM feedbacks to the prompt
        full_prompt = (
            f"{base_prompt}\n\n"
            "Two LLMs have provided feedback on the student's solution just for your reference:\n"
            f"1. Feedback from LLM 1 (Base Model):\n{base_feedback}\n\n"
            f"2. Feedback from LLM 2 (Llama Model):\n{llama_feedback}\n\n"
            "Your feedback:"
        )

        # Create chat template structure
        messages = [
            {"role": "user", "content": full_prompt}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    
    def generate_ta_feedback(self, question: str, ta_solution: str, stu_solution: str, 
                           base_feedback: str, llama_feedback: str, 
                           max_prompt_tokens: int = 4000) -> Dict[str, str]:
        inputs_dict = {
            "question": question,
            "ta_solution": ta_solution,
            "stu_solution": stu_solution,
            "base_feedback": base_feedback,
            "llama_feedback": llama_feedback
        }

        # Initial prompt creation
        prompt = self._create_ta_prompt(**inputs_dict)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]

        # Truncation loop if needed
        while len(input_ids) > max_prompt_tokens:
            print(f"Warning: Prompt exceeds {max_prompt_tokens} tokens ({len(input_ids)}). Truncating longest field.")
            
            # Find the longest field
            longest_field = max(inputs_dict, key=lambda k: len(inputs_dict[k]))
            
            # Truncate the longest field (remove 10% or min 100 chars from the end)
            trunc_len = max(100, int(len(inputs_dict[longest_field]) * 0.1))
            inputs_dict[longest_field] = inputs_dict[longest_field][:-trunc_len] + "..."
            
            # Recreate prompt and tokenize
            prompt = self._create_ta_prompt(**inputs_dict)
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]

            # Safety break in case of unexpected loop
            if len(inputs_dict[longest_field]) < 20: # Stop if a field becomes too small
                 print(f"Warning: Field '{longest_field}' became too short during truncation. Stopping truncation.")
                 break

        # Final tokenization for generation
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=1024,
                # temperature=0.7,
                # top_p=0.9,
                # do_sample=True,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = re.sub(r'^[Aa]ssistant\s*\n+', '', response, count=1).strip()
        # print("Full response:", response[-50:])  # Debug print
        
        # Extract the feedback text
        # The feedback should be everything after "Your feedback:"
        if "Your feedback:" in response:
            feedback = response.split("Your feedback:")[1].strip()
        else:
            # If we can't find the marker, use the last part of the response
            feedback = response.split("Your feedback:")[-1].strip()
        
        # If we still don't have feedback, use a default message
        if not feedback:
            feedback = "No feedback generated. Please try again."
        
        return {
            "ta_feedback": feedback
        }

def process_ta_feedback(input_file: str, output_file: str, model_name: str, max_prompt_tokens: int):
    # Load the questions and feedbacks
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Initialize the TA feedback simulator
    simulator = TAFeedbackSimulator(model_name=model_name)
    
    # Initialize an empty list for results (overwriting existing file)
    results = []
    
    # Process each question with progress bar
    for item in tqdm(data, desc="Generating TA feedback"):
        question_id = item['id']
        
        try:
            # Generate TA feedback
            ta_result = simulator.generate_ta_feedback(
                question=item['question_text'],
                ta_solution=item['ta_solution'],
                stu_solution=item['stu_solution'],
                base_feedback=item['base_feedback'],
                llama_feedback=item['llama_feedback'],
                max_prompt_tokens=max_prompt_tokens
            )
            
            # Create result dictionary
            result = {
                'id': question_id,
                'question_text': item['question_text'],
                'ta_solution': item['ta_solution'],
                'stu_solution': item['stu_solution'],
                'base_feedback': item['base_feedback'],
                'llama_feedback': item['llama_feedback'],
                'ta_feedback': ta_result['ta_feedback']
            }
            
            # Add to results
            results.append(result)
            
            # Save after each question in case of interruption
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error processing question {question_id}: {str(e)}")
            continue
    
    print(f"All TA feedback generated and saved to {output_file}")
    
    del simulator.model
    del simulator.tokenizer
    torch.cuda.empty_cache()
    import gc
    gc.collect()