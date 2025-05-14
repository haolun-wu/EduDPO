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
            device_map="auto",
            attn_implementation="flash_attention_2" if self.device == "cuda" else "eager",  # Use flash attention if available
        )
        
        # Set pad token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def _create_prompt(self, question: str, ta_solution: str, stu_solution: str) -> list:
        # Get the base prompt with the main variables
        base_prompt = PROMPT_TEMPLATE.format(
            question_text=question,
            ta_solution=ta_solution,
            stu_solution=stu_solution
        )

        # Add the LLM feedbacks to the prompt
        full_prompt = (
            f"{base_prompt}\n\n"
            "Your feedback:"
        )

        # Create chat template structure
        messages = [
            {"role": "user", "content": full_prompt}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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
                pad_token_id=self.tokenizer.pad_token_id,
                max_new_tokens=1024,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract feedback after "Your feedback:"
        if "Your feedback:" in response:
            feedback = response.split("Your feedback:")[1].strip()
        else:
            feedback = response.strip()
        
        feedback = re.sub(r'^\s*(<\|assistant\|>|\b[Aa]ssistant\b)\s*:?\s*\n*', '', feedback, count=1).strip()   
            
        return feedback

def process_llm_feedback(input_file: str, output_file: str, model_names: List[str]):
    # Load the questions and solutions
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    if len(model_names) != 2:
        raise ValueError("Exactly two model names must be provided: first for base_feedback, second for llama_feedback")
    
    # Initialize feedback generators for each model
    base_generator = FeedbackGenerator(model_name=model_names[0])
    llama_generator = FeedbackGenerator(model_name=model_names[1])
    
    # Process each question with progress bar
    results = []
    for item in tqdm(data, desc="Generating LLM feedbacks"):
        try:
            # Get feedback from both models
            base_feedback = base_generator.generate_feedback(
                question=item['question_text'],
                ta_solution=item['ta_solution'],
                stu_solution=item['stu_solution']
            )
            
            llama_feedback = llama_generator.generate_feedback(
                question=item['question_text'],
                ta_solution=item['ta_solution'],
                stu_solution=item['stu_solution']
            )
            
            # Create result dictionary with both feedbacks
            results.append({
                'id': item['id'],
                'question_text': item['question_text'],
                'ta_solution': item['ta_solution'],
                'stu_solution': item['stu_solution'],
                'base_feedback': base_feedback,
                'llama_feedback': llama_feedback
            })
            
            # Save after each question in case of interruption
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error processing question {item['id']}: {str(e)}")
            continue
    
    print(f"\nResults saved to {output_file}")
    
    del base_generator.model
    del llama_generator.model
    del base_generator.tokenizer
    del llama_generator.tokenizer
    torch.cuda.empty_cache()
    import gc
    gc.collect()