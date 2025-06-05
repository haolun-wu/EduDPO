import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import argparse
from scipy import stats
from typing import List, Dict, Tuple

AVAILABLE_MODELS = ['dpo', 'rpo', 'sft_v1', 'slic_beta_0.01', 'slic_beta_0.05']

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze correlation between edit distance and ROUGE scores')
    parser.add_argument('--models', type=str, nargs='+', default=AVAILABLE_MODELS,
                      choices=AVAILABLE_MODELS,
                      help='Models to analyze (default: all available models)')
    return parser.parse_args()

def load_edit_distances(edit_distance_file: str) -> dict:
    """Load edit distances from the saved JSON file."""
    with open(edit_distance_file, 'r') as f:
        return json.load(f)

def load_rouge_scores(judge_file: str) -> list:
    """Load ROUGE scores from the judge file."""
    rouge_scores = []
    with open(judge_file, 'r') as f:
        try:
            # First try to load the entire file
            data = json.load(f)
            rouge_scores = [item['rouge'] for item in data if 'rouge' in item]
        except json.JSONDecodeError:
            # If that fails, try to read line by line
            f.seek(0)
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
                            if 'rouge' in item:
                                rouge_scores.append(item['rouge'])
                except json.JSONDecodeError:
                    continue
    return rouge_scores

def plot_correlation(edit_distances: dict, rouge_scores: list, model_name: str, save_dir: str = None) -> Tuple[float, float, float]:
    """
    Create scatter plots showing correlation between edit distance and ROUGE scores.
    Each data point contributes two dots to the plot.
    Shows both normalized and raw edit distances in side-by-side subplots.
    Returns correlation coefficients for base, llama, and all points.
    """
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create edit_distance subdirectory
    save_dir = os.path.join(save_dir, 'edit_distance')
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Get distances
    normalized_distances = edit_distances['normalized_distances']
    raw_distances = edit_distances['raw_distances']
    
    # Separate distances for base and llama
    base_norm_distances = []
    llama_norm_distances = []
    base_raw_distances = []
    llama_raw_distances = []
    base_rouge = []
    llama_rouge = []
    
    # Process data points
    for i in range(0, len(normalized_distances), 2):
        if i + 1 < len(normalized_distances):
            # Get the ROUGE score for this sample
            rouge_score = rouge_scores[i // 2]
            
            # Store distances and scores
            base_norm_distances.append(normalized_distances[i])
            llama_norm_distances.append(normalized_distances[i + 1])
            base_raw_distances.append(raw_distances[i])
            llama_raw_distances.append(raw_distances[i + 1])
            base_rouge.append(rouge_score)
            llama_rouge.append(rouge_score)
    
    def plot_subplot(ax, x_base, x_llama, y, title, xlabel):
        # Plot points
        ax.scatter(x_base, y, color='blue', alpha=0.6, label='Base')
        ax.scatter(x_llama, y, color='red', alpha=0.6, label='Llama')
        
        # Calculate correlations
        base_corr = np.corrcoef(x_base, y)[0, 1]
        llama_corr = np.corrcoef(x_llama, y)[0, 1]
        all_corr = np.corrcoef(x_base + x_llama, y + y)[0, 1]
        
        # Calculate and plot trend lines
        def get_trend_line(x, y):
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            x_line = np.array([min(x), max(x)])
            y_line = slope * x_line + intercept
            return x_line, y_line
        
        x_base_line, y_base_line = get_trend_line(x_base, y)
        x_llama_line, y_llama_line = get_trend_line(x_llama, y)
        x_all_line, y_all_line = get_trend_line(x_base + x_llama, y + y)
        
        ax.plot(x_base_line, y_base_line, 'b--', alpha=0.5, label=f'Base trend (r={base_corr:.3f})')
        ax.plot(x_llama_line, y_llama_line, 'r--', alpha=0.5, label=f'Llama trend (r={llama_corr:.3f})')
        ax.plot(x_all_line, y_all_line, 'k--', alpha=0.5, label=f'All trend (r={all_corr:.3f})')
        
        # Add labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel('ROUGE Score')
        ax.set_title(title)
        
        # Add legend
        ax.legend()
        
        # Add correlation coefficients
        ax.text(0.05, 0.95, 
                f'Correlations:\nBase: {base_corr:.3f}\nLlama: {llama_corr:.3f}\nAll: {all_corr:.3f}', 
                transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        return base_corr, llama_corr, all_corr
    
    # Plot normalized distances
    base_norm_corr, llama_norm_corr, all_norm_corr = plot_subplot(
        ax1, base_norm_distances, llama_norm_distances, base_rouge,
        f'Normalized Edit Distance vs ROUGE Score ({model_name})',
        'Normalized Edit Distance'
    )
    
    # Plot raw distances
    base_raw_corr, llama_raw_corr, all_raw_corr = plot_subplot(
        ax2, base_raw_distances, llama_raw_distances, base_rouge,
        f'Raw Edit Distance vs ROUGE Score ({model_name})',
        'Raw Edit Distance'
    )
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'edit_distance_rouge_correlation_{model_name}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return normalized correlations (keeping original return values for compatibility)
    return base_norm_corr, llama_norm_corr, all_norm_corr

def plot_combined_correlations(correlations: Dict[str, Dict[str, float]], save_dir: str = None):
    """Create a bar plot comparing correlations across all models."""
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))
    
    save_dir = os.path.join(save_dir, 'edit_distance')
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 8))
    
    # Prepare data for plotting
    models = list(correlations.keys())
    base_corrs = [correlations[m]['base'] for m in models]
    llama_corrs = [correlations[m]['llama'] for m in models]
    all_corrs = [correlations[m]['all'] for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    # Create bars
    plt.bar(x - width, base_corrs, width, label='Base', color='blue', alpha=0.6)
    plt.bar(x, llama_corrs, width, label='Llama', color='red', alpha=0.6)
    plt.bar(x + width, all_corrs, width, label='All', color='black', alpha=0.6)
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Correlation Coefficient')
    plt.title(f'Correlation Coefficients Across Models ({", ".join(models)})')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    
    # Add correlation values on top of bars
    for i, v in enumerate(base_corrs):
        plt.text(i - width, v, f'{v:.3f}', ha='center', va='bottom')
    for i, v in enumerate(llama_corrs):
        plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    for i, v in enumerate(all_corrs):
        plt.text(i + width, v, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'combined_correlations.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()
    
    # File paths
    edit_distance_file = "src/analysis/edit_distance/edit_distances.json"
    
    # Load edit distances once
    edit_distances = load_edit_distances(edit_distance_file)
    
    # Store correlations for all models
    correlations = {}
    
    # Process each model
    for model in args.models:
        print(f"\nProcessing {model} model...")
        
        # Load ROUGE scores for this model
        judge_file = f"data/judge/gpt-4.1-mini_{model}_questions_ta_feedbacks_train.json"
        if not os.path.exists(judge_file):
            print(f"Warning: Judge file not found: {judge_file}")
            continue
            
        rouge_scores = load_rouge_scores(judge_file)
        
        # Plot correlation and get correlation coefficients
        base_corr, llama_corr, all_corr = plot_correlation(edit_distances, rouge_scores, model)
        
        # Store correlations
        correlations[model] = {
            'base': base_corr,
            'llama': llama_corr,
            'all': all_corr
        }
    
    # Plot combined correlations
    if correlations:
        plot_combined_correlations(correlations)
        print("\nAnalysis complete! Check the edit_distance directory for plots.")
    else:
        print("\nNo valid correlations were generated. Please check the input files.")

if __name__ == "__main__":
    main() 