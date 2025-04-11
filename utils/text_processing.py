import random
import re
from typing import Optional, Union, List, Dict
from dataclasses import dataclass

@dataclass
class TextModificationConfig:
    """Configuration for text modification parameters."""
    removal_probability: float = 0.3
    number_modification_range: float = 0.2
    number_modification_chance: float = 0.5
    decimal_places: int = 2

def add_random_modifications(text: str) -> str:
    """
    Add random modifications to the question text.
    
    Args:
        text: The input text to modify
        
    Returns:
        The modified text with random additions
    """
    modifications = [
        "Assume all events are independent",
        "Consider the following scenario",
        "Given these conditions",
        "Under these circumstances",
        "Taking into account",
        "With these parameters",
        "In this situation",
        "For this case",
        "Under these assumptions",
        "Given these constraints"
    ]
    
    # Randomly decide whether to add a modification
    if random.random() < 0.7:  # 70% chance to add a modification
        # Choose a random modification
        modification = random.choice(modifications)
        # Randomly decide where to insert it
        parts = text.split(',')
        if len(parts) > 1:
            insert_pos = random.randint(1, len(parts) - 1)
            parts.insert(insert_pos, modification)
            return ', '.join(parts)
        else:
            return f"{text}, {modification}"
    return text

def modify_numbers(
    text: str,
    modification_range: float = 0.2,
    modification_chance: float = 0.5,
    decimal_places: int = 2
) -> str:
    """
    Randomly modify numbers in the text by adding or subtracting a percentage.
    
    Args:
        text: The input text containing numbers to modify
        modification_range: The maximum percentage change (e.g., 0.2 for ±20%)
        modification_chance: Probability of modifying each number (0.0 to 1.0)
        decimal_places: Number of decimal places to round to for decimal numbers
        
    Returns:
        The text with randomly modified numbers
    """
    def modify_number(match: re.Match) -> str:
        try:
            num = float(match.group())
            if random.random() < modification_chance:
                modification = random.uniform(-modification_range, modification_range)
                new_num = num * (1 + modification)
                if num.is_integer():
                    return str(int(round(new_num)))
                return f"{new_num:.{decimal_places}f}"
            return match.group()
        except (ValueError, TypeError):
            return match.group()

    # Pattern to match numbers including decimals and negative numbers
    pattern = r'-?\d+\.?\d*'
    return re.sub(pattern, modify_number, text)

def split_into_parts(text: str) -> List[str]:
    """
    Split text into parts using commas and other common separators.
    
    Args:
        text: The input text to split
        
    Returns:
        List of text parts
    """
    # Split on commas, semicolons, and conjunctions
    parts = re.split(r'[,;]|\s(?:and|or|but)\s', text)
    # Remove empty strings and strip whitespace
    return [p.strip() for p in parts if p.strip()]

def randomly_remove_information(
    text: str,
    config: Optional[TextModificationConfig] = None,
    keep_original: bool = False
) -> str:
    """
    Randomly modify text by removing parts, changing numbers, and adding modifications.
    
    Args:
        text: The input text to modify
        config: Configuration for modification parameters. If None, uses defaults.
        keep_original: If True, returns the original text without modifications
        
    Returns:
        The modified text
    """
    if keep_original:
        return text
        
    if config is None:
        config = TextModificationConfig()
    
    # First modify numbers
    text = modify_numbers(
        text,
        modification_range=config.number_modification_range,
        modification_chance=config.number_modification_chance,
        decimal_places=config.decimal_places
    )
    
    # Then remove random parts
    parts = split_into_parts(text)
    filtered_parts = [
        p for p in parts 
        if random.random() > config.removal_probability
    ]
    
    # Join parts with commas
    text = ', '.join(filtered_parts)
    
    # Add random modifications
    text = add_random_modifications(text)
    
    return text 

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