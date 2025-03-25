#!/usr/bin/env python
"""
Basic usage examples for the PyRonounce package.
"""

import pyronounce
from pyronounce import PronounceabilityAssessor
import json

def example_single_word():
    """Example of assessing a single word."""
    print("\n=== Single Word Assessment ===")
    
    # Using the default assessor through the package function
    result = pyronounce.assess_word("supercalifragilisticexpialidocious")
    print(f"Word: {result['word']}")
    print(f"Score: {result['score']:.2f}")
    print(f"Category: {result['category']}")
    print(f"IPA: {result['ipa']}")
    
    # Get detailed feature information
    detailed = pyronounce.assess_word("antidisestablishmentarianism", detailed=True)
    print("\nDetailed assessment:")
    print(f"Word: {detailed['word']}")
    print(f"Score: {detailed['score']:.2f}")
    print(f"Category: {detailed['category']}")
    print(f"IPA: {detailed['ipa']}")
    print("Features:")
    for feature, value in detailed['features'].items():
        print(f"  {feature}: {value:.2f}")

def example_text_assessment():
    """Example of assessing a text sample."""
    print("\n=== Text Assessment ===")
    
    text = """The quick brown fox jumps over the lazy dog. 
    Pneumonoultramicroscopicsilicovolcanoconiosis is considered 
    one of the longest English words."""
    
    result = pyronounce.assess_text(text)
    print(f"Text: {result['text']}")
    print(f"Average score: {result['average_score']:.2f}")
    print(f"Overall category: {result['overall_category']}")
    print(f"Word count: {result['word_count']}")
    print(f"Words successfully assessed: {result['assessed_word_count']}")
    
    print("\nSample of word assessments:")
    for i, word in enumerate(result['words'][:3]):  # Show first 3 words
        if word['score'] is not None:
            print(f"  {word['word']}: {word['category']} ({word['score']:.2f})")
        else:
            print(f"  {word['word']}: Error - {word['error']}")

def example_custom_assessor():
    """Example of creating a custom assessor instance."""
    print("\n=== Custom Assessor ===")
    
    # Create a custom assessor instance
    assessor = PronounceabilityAssessor()
    
    # Get feature importance
    importance = assessor.get_feature_importance()
    print("Feature importance:")
    for feature, value in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {value:.2f}")
    
    # Compare several words
    words = ["simple", "complex", "difficult", "sixths"]
    print("\nComparing multiple words:")
    for word in words:
        result = assessor.assess_word(word)
        print(f"  {result['word']}: {result['category']} ({result['score']:.2f})")

def export_json_example():
    """Example of exporting assessment results to JSON."""
    print("\n=== JSON Export ===")
    
    words = ["simple", "complicated", "difficult", "supercalifragilisticexpialidocious"]
    results = [pyronounce.assess_word(word, detailed=True) for word in words]
    
    json_str = json.dumps(results, indent=2)
    print(f"JSON output (sample):\n{json_str[:200]}...")

if __name__ == "__main__":
    print("PyRonounce Usage Examples")
    print("-------------------------")
    example_single_word()
    example_text_assessment()
    example_custom_assessor()
    export_json_example() 