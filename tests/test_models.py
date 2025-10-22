"""
Unit tests for ML models module.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import FatigueClassifier


class TestFatigueClassifier(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Generate simple test data
        np.random.seed(42)
        self.X_train = pd.DataFrame(np.random.randn(100, 5), columns=[f'feat_{i}' for i in range(5)])
        self.y_train = np.random.randint(0, 2, 100)
        self.X_test = pd.DataFrame(np.random.randn(20, 5), columns=[f'feat_{i}' for i in range(5)])
        self.y_test = np.random.randint(0, 2, 20)
    
    def test_knn_initialization(self):
        """Test KNN classifier initialization."""
        classifier = FatigueClassifier(model_type='knn')
        self.assertEqual(classifier.model_type, 'knn')
        self.assertFalse(classifier.is_fitted)
    
    def test_svm_initialization(self):
        """Test SVM classifier initialization."""
        classifier = FatigueClassifier(model_type='svm')
        self.assertEqual(classifier.model_type, 'svm')
        self.assertFalse(classifier.is_fitted)
    
    def test_logistic_initialization(self):
        """Test Logistic Regression initialization."""
        classifier = FatigueClassifier(model_type='logistic')
        self.assertEqual(classifier.model_type, 'logistic')
        self.assertFalse(classifier.is_fitted)
    
    def test_invalid_model_type(self):
        """Test initialization with invalid model type."""
        with self.assertRaises(ValueError):
            FatigueClassifier(model_type='invalid')
    
    def test_train_knn(self):
        """Test training KNN classifier."""
        classifier = FatigueClassifier(model_type='knn')
        classifier.train(self.X_train, self.y_train)
        self.assertTrue(classifier.is_fitted)
    
    def test_train_svm(self):
        """Test training SVM classifier."""
        classifier = FatigueClassifier(model_type='svm')
        classifier.train(self.X_train, self.y_train)
        self.assertTrue(classifier.is_fitted)
    
    def test_train_logistic(self):
        """Test training Logistic Regression."""
        classifier = FatigueClassifier(model_type='logistic')
        classifier.train(self.X_train, self.y_train)
        self.assertTrue(classifier.is_fitted)
    
    def test_predict_before_training(self):
        """Test prediction before training raises error."""
        classifier = FatigueClassifier(model_type='knn')
        with self.assertRaises(ValueError):
            classifier.predict(self.X_test)
    
    def test_predict_after_training(self):
        """Test prediction after training."""
        classifier = FatigueClassifier(model_type='knn')
        classifier.train(self.X_train, self.y_train)
        predictions = classifier.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(all(p in [0, 1] for p in predictions))
    
    def test_predict_proba(self):
        """Test probability prediction."""
        classifier = FatigueClassifier(model_type='knn')
        classifier.train(self.X_train, self.y_train)
        probas = classifier.predict_proba(self.X_test)
        
        self.assertEqual(probas.shape[0], len(self.X_test))
        self.assertEqual(probas.shape[1], 2)  # Binary classification
        self.assertTrue(np.allclose(probas.sum(axis=1), 1))  # Probabilities sum to 1
    
    def test_evaluate(self):
        """Test model evaluation."""
        classifier = FatigueClassifier(model_type='knn')
        classifier.train(self.X_train, self.y_train)
        metrics = classifier.evaluate(self.X_test, self.y_test)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('confusion_matrix', metrics)
        self.assertIn('classification_report', metrics)
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
    
    def test_cross_validate(self):
        """Test cross-validation."""
        classifier = FatigueClassifier(model_type='knn')
        cv_scores = classifier.cross_validate(self.X_train, self.y_train, cv=3)
        
        self.assertIn('mean_score', cv_scores)
        self.assertIn('std_score', cv_scores)
        self.assertIn('all_scores', cv_scores)
        self.assertGreaterEqual(cv_scores['mean_score'], 0)
        self.assertLessEqual(cv_scores['mean_score'], 1)
    
    def test_prepare_data_from_list(self):
        """Test data preparation from list of dicts."""
        classifier = FatigueClassifier(model_type='knn')
        features = [{'feat_0': 1, 'feat_1': 2}, {'feat_0': 3, 'feat_1': 4}]
        labels = [0, 1]
        
        X, y = classifier.prepare_data(features, labels)
        
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(len(X), 2)
        self.assertEqual(len(y), 2)


if __name__ == '__main__':
    unittest.main()
