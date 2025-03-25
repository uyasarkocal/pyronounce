"""
Core functionality for the pyronounce package.
Provides functions for assessing word pronounceability.
"""

import numpy as np
import os
import pickle

from .utils import word_to_ipa, extract_features

# Path to the pre-trained model
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'data', 'default_model.pkl')

class PronounceabilityAssessor:
    """
    Class for assessing the pronounceability of English words.
    
    The model uses phonetic features to predict how difficult a word is to pronounce,
    returning a score from 0.0 (very hard) to 1.0 (very easy).
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the PronounceabilityAssessor with a pre-trained model.
        
        Args:
            model_path (str, optional): Path to a custom model file. If None, 
                                       uses the default pre-trained model.
        """
        model_path = model_path or DEFAULT_MODEL_PATH
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.weights = model_data['weights']
                self.bias = model_data['bias']
                self.normalization_params = model_data['normalization_params']
        except (FileNotFoundError, pickle.PickleError):
            # Fallback to training a basic model
            from .model import train_perceptron
            self.weights, self.bias, self.normalization_params = train_perceptron()
    
    def assess_word(self, word, detailed=False):
        """
        Assess the pronounceability of a word.
        
        Args:
            word (str): The word to assess.
            detailed (bool): Whether to return detailed feature information.
            
        Returns:
            dict: Assessment results including pronounceability score, 
                 difficulty category, and optionally feature details.
        """
        try:
            ipa, stress_markers = word_to_ipa(word)
            features = extract_features(ipa, stress_markers)
            
            # Normalize features using the same parameters as training
            X_mean, X_std = self.normalization_params
            features_normalized = (features - X_mean) / (X_std + 1e-10)
            
            # Calculate probability with sigmoid
            activation = np.dot(self.weights, features_normalized) + self.bias
            probability = 1 / (1 + np.exp(-activation))
            
            # Determine difficulty category
            if probability > 0.85:
                category = "very easy"
            elif probability > 0.65:
                category = "easy"
            elif probability > 0.45:
                category = "moderate"
            elif probability > 0.25:
                category = "hard" 
            else:
                category = "very hard"
            
            result = {
                'word': word,
                'ipa': ipa,
                'score': float(probability),
                'category': category
            }
            
            if detailed:
                # Include feature information
                feature_names = [
                    "syllables", "consonant_cluster", "vowel_ratio", 
                    "consonant_complexity", "diphthongs", "stress",
                    "length", "unusual_sounds"
                ]
                
                result['features'] = {name: float(val) for name, val in zip(feature_names, features)}
            
            return result
            
        except Exception as e:
            return {
                'word': word,
                'error': str(e),
                'score': None,
                'category': None
            }
    
    def assess_text(self, text, detailed=False):
        """
        Assess the pronounceability of a text by analyzing individual words.
        
        Args:
            text (str): The text to analyze.
            detailed (bool): Whether to return detailed feature information.
            
        Returns:
            dict: Assessment results including average score and word-by-word analysis.
        """
        # Basic tokenization - split on whitespace and remove punctuation
        words = []
        for word in text.split():
            word = ''.join(c for c in word if c.isalnum())
            if word:
                words.append(word.lower())
        
        # Assess each word
        word_assessments = [self.assess_word(word, detailed) for word in words]
        
        # Calculate average score (ignoring None values)
        valid_scores = [assessment['score'] for assessment in word_assessments 
                       if assessment['score'] is not None]
        
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
        
        # Determine overall category
        if avg_score is not None:
            if avg_score > 0.85:
                overall_category = "very easy"
            elif avg_score > 0.65:
                overall_category = "easy"
            elif avg_score > 0.45:
                overall_category = "moderate"
            elif avg_score > 0.25:
                overall_category = "hard" 
            else:
                overall_category = "very hard"
        else:
            overall_category = None
        
        return {
            'text': text,
            'average_score': avg_score,
            'overall_category': overall_category,
            'word_count': len(words),
            'assessed_word_count': len(valid_scores),
            'words': word_assessments
        }
    
    def get_feature_importance(self):
        """
        Return the relative importance of each feature in the model.
        
        Returns:
            dict: Feature names mapped to their relative importance.
        """
        feature_names = [
            "syllables", "consonant_cluster", "vowel_ratio", 
            "consonant_complexity", "diphthongs", "stress",
            "length", "unusual_sounds"
        ]
        
        # Calculate absolute weight values
        abs_weights = np.abs(self.weights)
        
        # Normalize to get relative importance
        total = np.sum(abs_weights)
        if total > 0:
            importance = abs_weights / total
        else:
            importance = np.ones_like(abs_weights) / len(abs_weights)
        
        return {name: float(imp) for name, imp in zip(feature_names, importance)} 