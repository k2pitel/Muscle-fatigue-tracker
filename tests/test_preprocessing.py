"""
Unit tests for EMG preprocessing module.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import EMGPreprocessor


class TestEMGPreprocessor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = EMGPreprocessor(sampling_rate=1000)
        self.test_signal = np.sin(2 * np.pi * 50 * np.linspace(0, 1, 1000))
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        self.assertEqual(self.preprocessor.sampling_rate, 1000)
    
    def test_bandpass_filter(self):
        """Test bandpass filtering."""
        filtered = self.preprocessor.bandpass_filter(self.test_signal)
        self.assertEqual(len(filtered), len(self.test_signal))
        self.assertIsInstance(filtered, np.ndarray)
    
    def test_normalize(self):
        """Test signal normalization."""
        normalized = self.preprocessor.normalize(self.test_signal)
        self.assertAlmostEqual(np.mean(normalized), 0, places=10)
        self.assertAlmostEqual(np.std(normalized), 1, places=10)
    
    def test_normalize_constant_signal(self):
        """Test normalization with constant signal."""
        constant_signal = np.ones(100)
        normalized = self.preprocessor.normalize(constant_signal)
        self.assertTrue(np.all(normalized == 0))
    
    def test_segment_signal(self):
        """Test signal segmentation."""
        window_size = 100
        overlap = 0.5
        segments = self.preprocessor.segment_signal(self.test_signal, window_size, overlap)
        
        self.assertIsInstance(segments, list)
        self.assertTrue(len(segments) > 0)
        self.assertEqual(len(segments[0]), window_size)
    
    def test_segment_signal_no_overlap(self):
        """Test segmentation without overlap."""
        window_size = 100
        overlap = 0
        segments = self.preprocessor.segment_signal(self.test_signal, window_size, overlap)
        
        expected_segments = len(self.test_signal) // window_size
        self.assertEqual(len(segments), expected_segments)
    
    def test_preprocess_pipeline(self):
        """Test complete preprocessing pipeline."""
        processed = self.preprocessor.preprocess_pipeline(self.test_signal)
        self.assertEqual(len(processed), len(self.test_signal))
        self.assertIsInstance(processed, np.ndarray)
    
    def test_preprocess_pipeline_no_filter(self):
        """Test preprocessing without filtering."""
        processed = self.preprocessor.preprocess_pipeline(
            self.test_signal, 
            apply_filter=False
        )
        self.assertIsInstance(processed, np.ndarray)
    
    def test_preprocess_pipeline_no_normalization(self):
        """Test preprocessing without normalization."""
        processed = self.preprocessor.preprocess_pipeline(
            self.test_signal, 
            apply_normalization=False
        )
        self.assertIsInstance(processed, np.ndarray)


if __name__ == '__main__':
    unittest.main()
