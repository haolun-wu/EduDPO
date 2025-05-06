import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from typing import List, Dict, Tuple
from pathlib import Path
import seaborn as sns
import os

class EditDistanceAnalyzer:
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize the analyzer with a tokenizer.
        
        Args:
            model_name: The name of the model to use for tokenization (default: "gpt2")
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set max length to handle longer sequences
        self.tokenizer.model_max_length = 2048
        
    def calculate_edit_distance(self, text1: str, text2: str) -> Tuple[int, float]:
        """
        Calculate both raw and normalized edit distance between two texts.
        Normalized distance is calculated as raw_distance / max(len(text1), len(text2))
        to ensure the value is in [0, 1].
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Tuple of (raw_edit_distance, normalized_edit_distance)
        """
        # Tokenize both texts with truncation
        tokens1 = self.tokenizer.encode(text1, truncation=True, max_length=2048)
        tokens2 = self.tokenizer.encode(text2, truncation=True, max_length=2048)
        
        # Calculate raw edit distance
        m, n = len(tokens1), len(tokens2)
        dp = np.zeros((m + 1, n + 1), dtype=int)
        
        # Initialize first row and column
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        # Fill the dp table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if tokens1[i-1] == tokens2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j-1] + 1,  # substitution
                                 dp[i-1][j] + 1,      # deletion
                                 dp[i][j-1] + 1)      # insertion
        
        raw_distance = dp[m][n]
        # Normalize by the maximum length of the two sequences
        normalized_distance = raw_distance / max(m, n) if max(m, n) > 0 else 0
        
        return raw_distance, normalized_distance
    
    def analyze_dpo_samples(self, dpo_file: str) -> Dict[str, List[float]]:
        """
        Analyze edit distances for all DPO samples.
        
        Args:
            dpo_file: Path to the DPO samples JSON file
            
        Returns:
            Dictionary containing raw and normalized distances for each sample
        """
        with open(dpo_file, 'r') as f:
            dpo_samples = json.load(f)
            
        raw_distances = []
        normalized_distances = []
        sample_ids = []
        
        for sample in dpo_samples:
            raw_dist, norm_dist = self.calculate_edit_distance(
                sample['chosen'],
                sample['rejected']
            )
            raw_distances.append(raw_dist)
            normalized_distances.append(norm_dist)
            sample_ids.append(sample['sample_id'])
            
        return {
            'raw_distances': raw_distances,
            'normalized_distances': normalized_distances,
            'sample_ids': sample_ids
        }
    
    def plot_distributions(self, distances: Dict[str, List[float]], save_dir: str = None):
        """
        Plot and save distributions of edit distances.
        
        Args:
            distances: Dictionary containing raw and normalized distances
            save_dir: Directory to save the plots (defaults to same directory as this file)
        """
        if save_dir is None:
            # Get the directory of the current file
            save_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create edit_distance subdirectory
        save_dir = os.path.join(save_dir, 'edit_distance')
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Set style using seaborn's default style
        sns.set_theme()
        
        # Plot raw distances
        plt.figure(figsize=(10, 6))
        sns.histplot(distances['raw_distances'], bins=50)
        plt.title('Distribution of Raw Edit Distances')
        plt.xlabel('Edit Distance')
        plt.ylabel('Count')
        plt.savefig(os.path.join(save_dir, 'raw_edit_distances.png'))
        plt.close()
        
        # Plot normalized distances
        plt.figure(figsize=(10, 6))
        sns.histplot(distances['normalized_distances'], bins=50)
        plt.title('Distribution of Normalized Edit Distances')
        plt.xlabel('Normalized Edit Distance')
        plt.ylabel('Count')
        plt.savefig(os.path.join(save_dir, 'normalized_edit_distances.png'))
        plt.close()
        
        # Convert to numpy arrays and save
        raw_distances_array = np.array(distances['raw_distances'])
        normalized_distances_array = np.array(distances['normalized_distances'])
        sample_ids_array = np.array(distances['sample_ids'])
        
        np.save(os.path.join(save_dir, 'raw_distances.npy'), raw_distances_array)
        np.save(os.path.join(save_dir, 'normalized_distances.npy'), normalized_distances_array)
        np.save(os.path.join(save_dir, 'sample_ids.npy'), sample_ids_array)
        
        # Save as a JSON file for easier inspection
        results = {
            'sample_ids': sample_ids_array.tolist(),  # Convert to Python list
            'raw_distances': raw_distances_array.tolist(),  # Convert to Python list
            'normalized_distances': normalized_distances_array.tolist()  # Convert to Python list
        }
        with open(os.path.join(save_dir, 'edit_distances.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
    def print_statistics(self, distances: Dict[str, List[float]]):
        """
        Print statistical information about the edit distances.
        
        Args:
            distances: Dictionary containing raw and normalized distances
        """
        raw_distances = distances['raw_distances']
        normalized_distances = distances['normalized_distances']
        
        print("\nRaw Edit Distance Statistics:")
        print(f"Mean: {np.mean(raw_distances):.2f}")
        print(f"Median: {np.median(raw_distances):.2f}")
        print(f"Std: {np.std(raw_distances):.2f}")
        print(f"Min: {np.min(raw_distances):.2f}")
        print(f"Max: {np.max(raw_distances):.2f}")
        
        print("\nNormalized Edit Distance Statistics:")
        print(f"Mean: {np.mean(normalized_distances):.2f}")
        print(f"Median: {np.median(normalized_distances):.2f}")
        print(f"Std: {np.std(normalized_distances):.2f}")
        print(f"Min: {np.min(normalized_distances):.2f}")
        print(f"Max: {np.max(normalized_distances):.2f}")

def main():
    # Initialize analyzer
    analyzer = EditDistanceAnalyzer()
    
    # Analyze DPO samples
    dpo_file = "data/processed/dpo_training_samples.json"
    distances = analyzer.analyze_dpo_samples(dpo_file)
    
    # Print statistics
    analyzer.print_statistics(distances)
    
    # Plot and save distributions
    analyzer.plot_distributions(distances)

if __name__ == "__main__":
    main() 