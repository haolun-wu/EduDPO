import json
from typing import List, Dict
from utils.text_processing import clean_text_formatting
# from ..config.prompts import generate_DPO_training_prompt


NEW2_AI_PROMPT = """
Focus on these issues only:
1) No explanation or solving the wrong problem.
2) Only an equation is provided without explanation (variable names/comments in code count as explanations).
3) Rare, but a major error in probability theory.

If no issues, give affirmative feedback (e.g., "Your learning shines!") in one phrase or sentence.

For issues, list them and provide brief corrective guidance if explanations were given.

Tone: Be a smart, kind, and insightful 20-year-old university teacher. Direct feedback to the student ("you").
"""

def generate_DPO_training_prompt(question_text: str, ta_solution: str, stu_solution: str) -> str:
    prompt = f'''
    You are providing feedback to students on their probability homework. 

    {NEW2_AI_PROMPT}
    
    Important: Return the feedback as a JSON dictionary. The dictionary should contain only two keys: "feedback" and "rubric".

    "feedback": A string providing feedback for the student's explanation.
    "rubric": One of the following strings: answer-correct, answer-incorrect, answer-major-error, answer-minor-error

    Information you'll need:
    Question: {question_text}
    Teacher solution: {ta_solution}
    The Student's Work: '''

    if stu_solution:
        prompt += f'''
    Student solution: {stu_solution}'''
    
    return clean_text_formatting(prompt)

def convert_to_dpo_samples(input_file: str, output_file: str) -> None:
    # Load the original data
    with open(input_file, 'r') as f:
        original_data = json.load(f)
    
    dpo_samples = []
    sample_id = 0
    
    for item in original_data:
        # Generate the prompt
        prompt = generate_DPO_training_prompt(
            question_text=item['question_text'],
            ta_solution=item['ta_solution'],
            stu_solution=item['stu_solution']
        )
        
        # Create two samples for each original item
        # Sample 1: chosen = ta_feedback, rejected = base_feedback
        dpo_samples.append({
            'sample_id': sample_id,
            'original_id': item['id'],
            'prompt': prompt,
            'chosen': clean_text_formatting(item['ta_feedback']),
            'rejected': clean_text_formatting(item['base_feedback'])
        })
        sample_id += 1
        
        # Sample 2: chosen = ta_feedback, rejected = llama_feedback
        dpo_samples.append({
            'sample_id': sample_id,
            'original_id': item['id'],
            'prompt': prompt,
            'chosen': clean_text_formatting(item['ta_feedback']),
            'rejected': clean_text_formatting(item['llama_feedback'])
        })
        sample_id += 1
    
    # Save the DPO samples
    with open(output_file, 'w') as f:
        json.dump(dpo_samples, f, indent=4, ensure_ascii=False)
