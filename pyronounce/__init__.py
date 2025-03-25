"""
Pronounceable: A package for assessing the pronounceability of English words.

This package provides tools to analyze how difficult words are to pronounce
based on their phonetic features.
"""

__version__ = "0.1.0"

from .core import PronounceabilityAssessor

# Create a default instance for easy access
assessor = PronounceabilityAssessor()

def assess_word(word, detailed=False):
    """
    Assess the pronounceability of a word.
    
    Args:
        word (str): The word to assess.
        detailed (bool): Whether to return detailed feature information.
        
    Returns:
        dict: Assessment results.
    """
    return assessor.assess_word(word, detailed)

def assess_text(text, detailed=False):
    """
    Assess the pronounceability of a text by analyzing individual words.
    
    Args:
        text (str): The text to analyze.
        detailed (bool): Whether to return detailed feature information.
        
    Returns:
        dict: Assessment results.
    """
    return assessor.assess_text(text, detailed) 