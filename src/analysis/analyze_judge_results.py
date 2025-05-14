import json
import glob
import os
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

def extract_model_name(filename: str) -> str:
    """Extract model name from filename."""
    base = os.path.basename(filename)
    # Extract text between 'judging_' and '_questions_ta_feedbacks'
    model_name = base.split('judging_')[1].split('_questions_ta_feedbacks')[0]
    return model_name

def get_model_priority(model_name: str) -> int:
    """Get priority for model sorting. Lower number means higher priority."""
    model_name = model_name.lower()
    if 'sft' in model_name:
        return 0
    elif 'dpo' in model_name:
        return 1
    elif 'rpo' in model_name:
        return 2
    else:
        return 3

def analyze_judge_file(filepath: str) -> Tuple[float, int, int, int, int]:
    """Analyze a single judge file and return average ROUGE, correctness count, helpfulness count, both_true_count, and total items."""
    total_rouge = 0
    correctness_count = 0
    helpfulness_count = 0
    both_true_count = 0  # Count of cases where both correctness and helpfulness are true
    count = 0
    
    # Read file line by line to handle potential JSON corruption
    with open(filepath, 'r') as f:
        try:
            # First try to load the entire file
            data = json.load(f)
            items = data
        except json.JSONDecodeError:
            # If that fails, try to read line by line
            f.seek(0)
            items = []
            for line in f:
                try:
                    if line.strip() and line.strip() != '[' and line.strip() != ']':
                        # Clean up the line to make it a valid JSON object
                        line = line.strip().rstrip(',')
                        if line.startswith('['):
                            line = line[1:]
                        if line.endswith(']'):
                            line = line[:-1]
                        if line:
                            item = json.loads(line)
                            items.append(item)
                except json.JSONDecodeError:
                    continue
    
    for item in items:
        if isinstance(item, dict):  # Make sure we have a dictionary
            if 'rouge' in item:
                total_rouge += item['rouge']
                count += 1
            is_correct = item.get('correctness', False)
            is_helpful = item.get('helpfulness', False)
            if is_correct:
                correctness_count += 1
            if is_helpful:
                helpfulness_count += 1
            if is_correct and is_helpful:  # Count cases where both are true
                both_true_count += 1
    
    avg_rouge = total_rouge / count if count > 0 else 0
    return avg_rouge, correctness_count, helpfulness_count, both_true_count, len(items)

def save_table_as_figure(df: pd.DataFrame, save_dir: str = None):
    """Save the results table as a figure."""
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create results subdirectory
    save_dir = os.path.join(save_dir, 'results')
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('default')  # Use default style instead of seaborn
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.5 + 2))
    
    # Hide axes
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Adjust table style
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='white')
        else:
            cell.set_facecolor('#f0f0f0' if row % 2 == 0 else 'white')
    
    # Add title
    plt.title('Judge Results Summary', pad=20, size=14, weight='bold')
    
    # Save figure
    plt.savefig(os.path.join(save_dir, 'judge_results_table.png'),
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.5)
    plt.close()

def main():
    # Get all judge files
    judge_files = glob.glob('data/judge/judging_*_questions_ta_feedbacks_*.json')
    
    if not judge_files:
        print("No judge files found in data/judge/ directory!")
        print("Current working directory:", os.getcwd())
        print("Trying alternative path...")
        # Try alternative path
        judge_files = glob.glob('../data/judge/judging_*_questions_ta_feedbacks_*.json')
        if not judge_files:
            print("No judge files found in ../data/judge/ directory either!")
            return
    
    print(f"Found {len(judge_files)} judge files:")
    for file in judge_files:
        print(f"  - {file}")
    
    # Initialize results dictionary
    results = []
    
    # Process each file
    for filepath in judge_files:
        try:
            model_name = extract_model_name(filepath)
            avg_rouge, correctness_count, helpfulness_count, both_true_count, total_items = analyze_judge_file(filepath)
            
            results.append({
                'Model': model_name,
                'Avg ROUGE Score': round(avg_rouge, 4),
                'Correctness=True': f"{correctness_count}/{total_items}",
                'Helpfulness=True': f"{helpfulness_count}/{total_items}",
                'Both=True': f"{both_true_count}/{total_items}"
            })
            print(f"Successfully processed {model_name}")
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
    
    if not results:
        print("No results were generated. Please check the judge files.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Add priority column for sorting
    df['priority'] = df['Model'].apply(get_model_priority)
    
    # Sort by priority first, then by Both=True count, and finally by ROUGE score
    df = df.sort_values(['priority', 'Both=True', 'Avg ROUGE Score'], ascending=[True, False, False])
    
    # Remove the priority column before displaying
    df = df.drop('priority', axis=1)
    
    # Print results
    print("\nJudge Results Summary:")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
    print("\nNote: Both=True shows the count of samples where both Correctness and Helpfulness are True")
    print("      Each correct feedback adds +1 to Correctness=True count")
    print("      Each helpful feedback adds +1 to Helpfulness=True count")
    print("      ROUGE Score measures the similarity between feedback texts")
    
    # Save table as figure
    save_table_as_figure(df)

if __name__ == "__main__":
    main() 