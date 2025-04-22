import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional
from utils.text_processing import clean_text_formatting

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
            low_cpu_mem_usage=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _create_ta_prompt(self, question: str, ta_solution: str, stu_solution: str, 
                         base_feedback: str, llama_feedback: str) -> str:
        return f"""You are an experienced and kind teaching assistant (TA) in a probability course. 
                    Your task is to provide detailed feedback on a student's solution to a probability problem. 
                    You should first state whether the student's solution is correct or not, and write a single paragraph of feedback.

                    Problem:
                    {question}

                    Suggested Solution (for reference):
                    {ta_solution}

                    Student's Solution:
                    {stu_solution}

                    Two LLMs have provided feedback on the student's solution just for your reference:
                    1. Feedback from LLM 1 (Base Model):
                    {base_feedback}

                    2. Feedback from LLM 2 (Llama Model):
                    {llama_feedback}

                    Your feedback: """
                    
    # def _create_ta_prompt(self, question: str, ta_solution: str, stu_solution: str, 
    #                     base_feedback: str, llama_feedback: str) -> str:
    #     user_message = f"""You are an experienced and kind teaching assistant (TA) in a probability course. 
    #             Your task is to provide detailed feedback on a student's solution to a probability problem. 
    #             You should first state whether the student's solution is correct or not, and write a single paragraph of feedback.

    #             Problem:
    #             {question}

    #             Suggested Solution (for reference):
    #             {ta_solution}

    #             Student's Solution:
    #             {stu_solution}

    #             Two LLMs have provided feedback on the student's solution just for your reference:
    #             1. Feedback from LLM 1 (Base Model):
    #             {base_feedback}

    #             2. Feedback from LLM 2 (Llama Model):
    #             {llama_feedback}

    #             Your feedback:"""

    #     return f"<|user|>{user_message}<|end|><|assistant|>"


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
        print("Input length: ", len(input_ids))

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
                max_new_tokens=1024,
                # temperature=0.7,
                # top_p=0.9,
                # do_sample=True,
                # pad_token_id=self.tokenizer.eos_token_id
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
    
    # Process each question
    for item in data:
        question_id = item['id']
        
        # # Check if we already have TA feedback for this question
        # existing_feedback = next((r for r in results if r['id'] == question_id), None)
        # if existing_feedback:
        #     print(f"Using existing TA feedback for question {question_id}")
        #     continue
        
        print(f"Generating TA feedback for question {question_id}")
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