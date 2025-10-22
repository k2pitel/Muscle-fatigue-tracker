"""
Machine learning models for muscle fatigue classification.
Implements KNN, SVM, and Logistic Regression classifiers.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


class FatigueClassifier:
    """
    Muscle fatigue classifier using multiple ML algorithms.
    """
    
    def __init__(self, model_type='knn'):
        """
        Initialize the classifier.
        
        Args:
            model_type (str): Type of model ('knn', 'svm', or 'logistic')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Initialize the selected model
        if model_type == 'knn':
            self.model = KNeighborsClassifier(n_neighbors=5)
        elif model_type == 'svm':
            self.model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        elif model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'knn', 'svm', or 'logistic'")
    
    def prepare_data(self, features, labels):
        """
        Prepare feature data for training.
        
        Args:
            features (list or pd.DataFrame): Feature data
            labels (list or np.ndarray): Target labels
            
        Returns:
            tuple: (X, y) prepared data
        """
        # Convert features to DataFrame if it's a list of dicts
        if isinstance(features, list) and isinstance(features[0], dict):
            X = pd.DataFrame(features)
        else:
            X = pd.DataFrame(features)
        
        y = np.array(labels)
        
        return X, y
    
    def train(self, X_train, y_train):
        """
        Train the classifier.
        
        Args:
            X_train (pd.DataFrame or np.ndarray): Training features
            y_train (np.ndarray): Training labels
            
        Returns:
            self: Trained classifier
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        return self
    
    def predict(self, X_test):
        """
        Make predictions on test data.
        
        Args:
            X_test (pd.DataFrame or np.ndarray): Test features
            
        Returns:
            np.ndarray: Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        
        return predictions
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities.
        
        Args:
            X_test (pd.DataFrame or np.ndarray): Test features
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_test_scaled = self.scaler.transform(X_test)
        
        # Check if model supports predict_proba
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_test_scaled)
            return probabilities
        else:
            raise AttributeError(f"Model {self.model_type} does not support probability predictions")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the classifier on test data.
        
        Args:
            X_test (pd.DataFrame or np.ndarray): Test features
            y_test (np.ndarray): True labels
            
        Returns:
            dict: Evaluation metrics
        """
        predictions = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        class_report = classification_report(y_test, predictions, output_dict=True)
        
        metrics = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation.
        
        Args:
            X (pd.DataFrame or np.ndarray): Features
            y (np.ndarray): Labels
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Cross-validation scores
        """
        X_scaled = self.scaler.fit_transform(X)
        scores = cross_val_score(self.model, X_scaled, y, cv=cv)
        
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'all_scores': scores
        }
    
    def tune_hyperparameters(self, X_train, y_train, param_grid=None, cv=5):
        """
        Tune hyperparameters using grid search.
        
        Args:
            X_train (pd.DataFrame or np.ndarray): Training features
            y_train (np.ndarray): Training labels
            param_grid (dict): Parameter grid for grid search
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Best parameters and best score
        """
        if param_grid is None:
            # Default parameter grids
            if self.model_type == 'knn':
                param_grid = {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance']
                }
            elif self.model_type == 'svm':
                param_grid = {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto'],
                    'kernel': ['rbf', 'linear']
                }
            elif self.model_type == 'logistic':
                param_grid = {
                    'C': [0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'liblinear']
                }
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        grid_search = GridSearchCV(self.model, param_grid, cv=cv, scoring='accuracy')
        grid_search.fit(X_train_scaled, y_train)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.is_fitted = True
