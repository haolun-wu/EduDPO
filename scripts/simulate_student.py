import argparse
from src import process_student_answers, TextModificationConfig

def main():
    parser = argparse.ArgumentParser(description='Generate student solutions using LLM')
    parser.add_argument('--input_file', type=str, default='data/raw/questions.json',
                        help='Input JSON file containing questions and TA solutions')
    parser.add_argument('--output_file', type=str, default='data/simulated/questions_stu_answers.json',
                        help='Output JSON file to save student solutions')
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-Instruct-v0.3',
                        help='Model to use for generating student solutions')
    parser.add_argument('--num_simulations', type=int, default=5,
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
    
    # Generate student answers
    print("Generating student answers...")
    process_student_answers(
        input_file=args.input_file,
        output_file=args.output_file,
        model_name=args.model,
        num_simulations=args.num_simulations,
        text_modification_config=config
    ) 

if __name__ == "__main__":
    main() 