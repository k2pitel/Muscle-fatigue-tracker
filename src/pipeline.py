"""
Training pipeline for muscle fatigue detection.
Orchestrates data preprocessing, feature extraction, and model training.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .preprocessing import EMGPreprocessor
from .feature_extraction import EMGFeatureExtractor
from .models import FatigueClassifier


class FatigueDetectionPipeline:
    """
    Complete pipeline for muscle fatigue detection.
    """
    
    def __init__(self, sampling_rate=1000, model_type='knn'):
        """
        Initialize the pipeline.
        
        Args:
            sampling_rate (int): Sampling rate of EMG signals
            model_type (str): Type of ML model ('knn', 'svm', or 'logistic')
        """
        self.sampling_rate = sampling_rate
        self.preprocessor = EMGPreprocessor(sampling_rate=sampling_rate)
        self.feature_extractor = EMGFeatureExtractor(sampling_rate=sampling_rate)
        self.classifier = FatigueClassifier(model_type=model_type)
        self.model_type = model_type
    
    def process_signal(self, signal_data, window_size=1000, overlap=0.5):
        """
        Process a single EMG signal through preprocessing and feature extraction.
        
        Args:
            signal_data (np.ndarray): Raw EMG signal
            window_size (int): Window size for segmentation
            overlap (float): Overlap ratio for segmentation
            
        Returns:
            list: Extracted features for each segment
        """
        # Preprocess the signal
        processed_signal = self.preprocessor.preprocess_pipeline(signal_data)
        
        # Segment the signal
        segments = self.preprocessor.segment_signal(processed_signal, window_size, overlap)
        
        # Extract features from segments
        features = self.feature_extractor.extract_features_from_segments(segments)
        
        return features
    
    def prepare_dataset(self, signals, labels, window_size=1000, overlap=0.5):
        """
        Prepare a complete dataset from multiple signals.
        
        Args:
            signals (list): List of EMG signals
            labels (list): Corresponding fatigue labels for each signal
            window_size (int): Window size for segmentation
            overlap (float): Overlap ratio for segmentation
            
        Returns:
            tuple: (features_df, labels_array)
        """
        all_features = []
        all_labels = []
        
        for signal_data, label in zip(signals, labels):
            # Process signal and extract features
            segment_features = self.process_signal(signal_data, window_size, overlap)
            
            # Add features and labels
            all_features.extend(segment_features)
            all_labels.extend([label] * len(segment_features))
        
        # Convert to DataFrame and array
        features_df = pd.DataFrame(all_features)
        labels_array = np.array(all_labels)
        
        return features_df, labels_array
    
    def train(self, X_train, y_train):
        """
        Train the classifier.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (np.ndarray): Training labels
            
        Returns:
            self: Trained pipeline
        """
        self.classifier.train(X_train, y_train)
        return self
    
    def predict(self, signal_data, window_size=1000, overlap=0.5):
        """
        Predict fatigue level for a new signal.
        
        Args:
            signal_data (np.ndarray): Raw EMG signal
            window_size (int): Window size for segmentation
            overlap (float): Overlap ratio for segmentation
            
        Returns:
            np.ndarray: Predicted labels for each segment
        """
        # Extract features
        features = self.process_signal(signal_data, window_size, overlap)
        features_df = pd.DataFrame(features)
        
        # Make predictions
        predictions = self.classifier.predict(features_df)
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the pipeline on test data.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (np.ndarray): True labels
            
        Returns:
            dict: Evaluation metrics
        """
        return self.classifier.evaluate(X_test, y_test)
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation.
        
        Args:
            X (pd.DataFrame): Features
            y (np.ndarray): Labels
            cv (int): Number of folds
            
        Returns:
            dict: Cross-validation scores
        """
        return self.classifier.cross_validate(X, y, cv=cv)
    
    def save(self, filepath):
        """
        Save the trained pipeline.
        
        Args:
            filepath (str): Path to save the model
        """
        self.classifier.save_model(filepath)
    
    def load(self, filepath):
        """
        Load a trained pipeline.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.classifier.load_model(filepath)


def train_multiple_models(X_train, y_train, X_test, y_test):
    """
    Train and compare multiple models (KNN, SVM, Logistic Regression).
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (np.ndarray): Training labels
        X_test (pd.DataFrame): Test features
        y_test (np.ndarray): Test labels
        
    Returns:
        dict: Results for each model
    """
    models = ['knn', 'svm', 'logistic']
    results = {}
    
    for model_type in models:
        print(f"\nTraining {model_type.upper()} model...")
        
        # Create and train classifier
        classifier = FatigueClassifier(model_type=model_type)
        classifier.train(X_train, y_train)
        
        # Evaluate
        metrics = classifier.evaluate(X_test, y_test)
        
        results[model_type] = {
            'classifier': classifier,
            'metrics': metrics
        }
        
        print(f"{model_type.upper()} Accuracy: {metrics['accuracy']:.4f}")
    
    return results
