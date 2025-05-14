import dspy 
from typing import Dict

NEW2_AI_PROMPT = """
Focus on these issues only:
1) No explanation or solving the wrong problem.
2) Only an equation is provided without explanation (variable names/comments in code count as explanations).
3) Rare, but a major error in probability theory.

If no issues, give affirmative feedback (e.g., "Your learning shines!") in one phrase or sentence.

For issues, list them and provide brief corrective guidance if explanations were given.

Tone: Be a smart, kind, and insightful 20-year-old university teacher. Direct feedback to the student ("you").
"""
    
class JudgingFeedbackSignature(dspy.Signature):
    """
    You are a helpful assistant for an introductory Python programming course.

    You are provided with:
    - A problem description (the question text)
    - The teacher solution to the problem
    - The student attempt at solving the problem 
    - The feedback written by another teaching assistant

    Your job is to evaluate whether the feedback is both correct and helpful.

    Begin by reflecting deeply on the quality of the written feedback. 
    Identify areas where the feedback is good and areas where the feedback could be improved.
    Explain your reasoning clearly and precisely.

    Then, provide your final assessment of the feedback along two dimensions: correcntess and helpfullness.
    Write this assessment in a JSON dictionary with two keys:
    "correctness": true or false (whether the feedback is correct)
    "helpfulness": true or false (whether the feedback is helpful)

    """

    question_text = dspy.InputField()
    ta_solution = dspy.InputField()
    stu_solution = dspy.InputField()
    feedback = dspy.InputField()

    reasoning: str = dspy.OutputField()
    asessment: Dict[str, bool] = dspy.OutputField()


def build_dspy_dataset(dataframe):
    dspy_dataset = []
    for row in dataframe.itertuples(index=False):
        dspy_dataset.append(dspy.Example(
            question_text=row.question_text,
            ta_solution=row.ta_solution,
            stu_solution=row.stu_solution,
            feedback=row.feedback,
        ).with_inputs("question_text", "ta_solution", "stu_solution", "feedback"))

    return dspy_dataset