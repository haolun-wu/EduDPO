import re

NEW2_AI_PROMPT = """
Focus on these issues only:
1) No explanation or solving the wrong problem.
2) Only an equation is provided without explanation (variable names/comments in code count as explanations).
3) Rare, but a major error in probability theory.

If no issues, give affirmative feedback in one phrase or sentence.

For issues, list them and provide brief corrective guidance if explanations were given.

You should first state whether the student's solution is correct or not.

Tone: Be a smart, kind, and insightful 20-year-old university teacher. Direct feedback to the student ("you").
"""

def generate_judging_prompt(question_text: str, ta_solution: str, stu_solution: str, feedback: str) -> str:
    prompt = f'''
    You are evaluating the quality of the feedback to a students' probability homework. 
    
    Bellow is the feedback task description given to the annotator:
    {NEW2_AI_PROMPT}
    
    Information you will also need:
    Question: {question_text}
    Teacher solution: {ta_solution}
    '''

    if stu_solution:
        prompt += f'''
    Student solution: {stu_solution}'''
        
    prompt += f'''
    Feedback: {feedback}

    Your task is to evaluate whether this feedback is correct and helpful.
    Important: Return your answer as a JSON dictionary. The dictionary should contain only two keys: "correctness" and "helpfulness".

    "correctness": true or false (whether the feedback is correct)
    "helpfulness": true or false (whether the feedback is helpful)

    '''
    
    return clean_text_formatting(prompt)


def clean_text_formatting(text: str) -> str:
    """
    Clean up text formatting by:
    1. Removing all newlines (\n)
    2. Removing extra spaces after newlines
    3. Ensuring proper spacing between sentences
    
    Args:
        text (str): The input text to clean
        
    Returns:
        str: The cleaned text with proper formatting
    """
    # First, replace all newlines with spaces
    text = text.replace('\n', ' ')
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Ensure proper spacing after periods (except for decimal points)
    text = re.sub(r'\.(\d)', r'.\1', text)  # Preserve decimal points
    text = re.sub(r'\.(\s*)', '. ', text)   # Ensure space after periods
    
    # Ensure proper spacing after other sentence-ending punctuation
    text = re.sub(r'!(\s*)', '! ', text)
    text = re.sub(r'\?(\s*)', '? ', text)
    
    # Remove any remaining multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing spaces
    text = text.strip()
    
    return text 

