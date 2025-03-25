"""
Model training functionality for the pyronounce package.
"""

import numpy as np
import os
import pickle
from .utils import word_to_ipa, extract_features

def train_perceptron(save_path=None):
    """
    Train a perceptron model to classify word pronounceability.
    
    Args:
        save_path (str, optional): Path to save the trained model.
        
    Returns:
        tuple: (weights, bias, normalization_params)
    """
    # Training data with balanced examples
    X_data = [
        # Easy words (1.0)
        ("cat", 1.0), ("dog", 1.0), ("fish", 1.0), ("book", 1.0), ("pen", 1.0),
        ("desk", 1.0), ("chair", 1.0), ("house", 1.0), ("tree", 1.0), ("ball", 1.0),
        ("hand", 1.0), ("head", 1.0), ("foot", 1.0), ("shoe", 1.0), ("door", 1.0),
        ("bird", 1.0), ("mouse", 1.0), ("day", 1.0), ("sun", 1.0), ("sky", 1.0),
        
        # Moderately easy words (0.8)
        ("paper", 0.8), ("remote", 0.8), ("lighter", 0.8), ("water", 0.8),
        ("apple", 0.8), ("baby", 0.8), ("window", 0.8), ("table", 0.8),
        ("mother", 0.8), ("father", 0.8), ("sister", 0.8), ("brother", 0.8),
        ("simple", 0.8), ("easy", 0.8), ("morning", 0.8), ("evening", 0.8),
        
        # Medium words (0.6)
        ("tobacco", 0.6), ("perfume", 0.6), ("incense", 0.6), ("terminal", 0.6),
        ("computer", 0.6), ("important", 0.6), ("tomorrow", 0.6), ("together", 0.6),
        ("chocolate", 0.6), ("saturday", 0.6), ("happiness", 0.6), ("beautiful", 0.6),
        ("diabetes", 0.5), ("business", 0.6), ("attention", 0.6), ("amazing", 0.6),
        
        # Moderately difficult words (0.4)
        ("pronunciation", 0.4), ("deliberately", 0.4), ("vocabulary", 0.4),
        ("particularly", 0.4), ("statistics", 0.4), ("psychology", 0.4),
        ("university", 0.4), ("mathematics", 0.4), ("dictionary", 0.4),
        ("laboratory", 0.4), ("technology", 0.4), ("unforgettable", 0.4),
        ("extraordinary", 0.4), ("zeitgeist", 0.3), ("squirrel", 0.35),
        
        # Difficult words (0.2)
        ("anemone", 0.2), ("phenomenon", 0.2), ("rhythmic", 0.2), ("xylophone", 0.15),
        ("chrysanthemum", 0.2), ("isthmus", 0.2), ("worcestershire", 0.2), ("rural", 0.2),
        ("anesthetist", 0.2), ("otorhinolaryngology", 0.2), ("specificity", 0.2),
        ("deliberation", 0.25), ("confabulation", 0.2), ("contemplation", 0.2),
        
        # Very difficult words (0.0)
        ("sixths", 0.0), ("strengths", 0.0), ("twelfths", 0.0), ("synecdoche", 0.0),
        ("tschüss", 0.0), ("pseudopseudohypoparathyroidism", 0.0), 
        ("pneumonoultramicroscopicsilicovolcanoconiosis", 0.0),
        ("complicated", 0.1), ("phantasy", 0.1)
    ]
    
    # Process words into features
    X = []
    y = []
    
    for word, label in X_data:
        try:
            ipa, stress_markers = word_to_ipa(word)
            features = extract_features(ipa, stress_markers)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"Error processing '{word}': {e}")
    
    X = np.array(X)
    y = np.array(y)
    
    # Apply feature scaling for better convergence
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_normalized = (X - X_mean) / (X_std + 1e-10)  # Avoid division by zero
    
    # Train a more robust model
    np.random.seed(42)
    weights = np.random.randn(X_normalized.shape[1])  # Match feature count
    bias = 0
    learning_rate = 0.05
    
    # Save normalization parameters for prediction
    normalization_params = (X_mean, X_std)
    
    for epoch in range(2000):  # More epochs for better convergence
        # Shuffle data each epoch
        indices = np.random.permutation(len(X_normalized))
        X_shuffled = X_normalized[indices]
        y_shuffled = y[indices]
        
        for xi, yi in zip(X_shuffled, y_shuffled):
            activation = np.dot(weights, xi) + bias
            prediction = 1 / (1 + np.exp(-activation))  # Sigmoid
            
            # Update weights with gradient
            error = yi - prediction
            weights += learning_rate * error * xi
            bias += learning_rate * error
    
    # Save the model if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model_data = {
            'weights': weights,
            'bias': bias,
            'normalization_params': normalization_params
        }
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    return weights, bias, normalization_params

def train_and_save_default_model():
    """
    Train and save the default model to the package data directory.
    This function is used during package installation.
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    model_path = os.path.join(data_dir, 'default_model.pkl')
    train_perceptron(save_path=model_path)
    return model_path 