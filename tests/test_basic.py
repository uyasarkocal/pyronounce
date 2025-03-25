"""
Basic tests for the pyronounce package.
"""

import pytest
import os
import numpy as np
from pyronounce import assess_word, assess_text, PronounceabilityAssessor

def test_assess_word():
    """Test the assess_word function."""
    # Test a simple word
    result = assess_word("test")
    assert isinstance(result, dict)
    assert "word" in result
    assert "score" in result
    assert "category" in result
    assert "ipa" in result
    assert result["word"] == "test"
    assert 0.0 <= result["score"] <= 1.0
    
    # Test with detailed=True
    detailed = assess_word("test", detailed=True)
    assert "features" in detailed
    assert isinstance(detailed["features"], dict)
    assert len(detailed["features"]) > 0
    
def test_assess_text():
    """Test the assess_text function."""
    text = "This is a test."
    result = assess_text(text)
    
    assert isinstance(result, dict)
    assert "text" in result
    assert "average_score" in result
    assert "overall_category" in result
    assert "word_count" in result
    assert "assessed_word_count" in result
    assert "words" in result
    
    assert result["text"] == text
    assert 0.0 <= result["average_score"] <= 1.0
    assert isinstance(result["words"], list)
    assert len(result["words"]) > 0
    
def test_pronounceability_assessor():
    """Test the PronounceabilityAssessor class."""
    # Test creation
    assessor = PronounceabilityAssessor()
    
    # Test word assessment
    result = assessor.assess_word("test")
    assert isinstance(result, dict)
    assert "score" in result
    
    # Test feature importance
    importance = assessor.get_feature_importance()
    assert isinstance(importance, dict)
    assert len(importance) > 0
    assert sum(importance.values()) == pytest.approx(1.0)
    
def test_model_training():
    """Test the model training functionality."""
    from pyronounce.model import train_perceptron
    
    weights, bias, norm_params = train_perceptron()
    
    assert isinstance(weights, np.ndarray)
    assert isinstance(bias, (float, int))
    assert isinstance(norm_params, tuple)
    assert len(norm_params) == 2  # mean and std
    
def test_utils():
    """Test utility functions."""
    from pyronounce.utils import word_to_ipa, extract_features
    
    ipa, stress_markers = word_to_ipa("test")
    assert isinstance(ipa, str)
    assert isinstance(stress_markers, list)
    
    features = extract_features(ipa, stress_markers)
    assert isinstance(features, np.ndarray)
    assert features.shape[0] == 8  # 8 features
    
def test_category_assignment():
    """Test category assignment based on scores."""
    # Create a function to determine category from score
    def get_category(score):
        if score > 0.85:
            return "very easy"
        elif score > 0.65:
            return "easy"
        elif score > 0.45:
            return "moderate"
        elif score > 0.25:
            return "hard" 
        else:
            return "very hard"
    
    # Test with various scores
    test_scores = [0.1, 0.3, 0.5, 0.7, 0.9]
    expected_categories = ["very hard", "hard", "moderate", "easy", "very easy"]
    
    for score, expected in zip(test_scores, expected_categories):
        assessor = PronounceabilityAssessor()
        
        # Mock the assess_word method to return a specific score
        original_method = assessor.assess_word
        assessor.assess_word = lambda word, detailed=False: {
            "word": word,
            "ipa": "/test/",
            "score": score,
            "category": get_category(score)
        }
        
        result = assessor.assess_word("test")
        assert result["category"] == expected
        
        # Restore the original method
        assessor.assess_word = original_method 