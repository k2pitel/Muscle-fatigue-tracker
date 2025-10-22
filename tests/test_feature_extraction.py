"""
Unit tests for feature extraction module.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.feature_extraction import EMGFeatureExtractor


class TestEMGFeatureExtractor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = EMGFeatureExtractor(sampling_rate=1000)
        self.test_signal = np.sin(2 * np.pi * 50 * np.linspace(0, 1, 1000))
    
    def test_initialization(self):
        """Test extractor initialization."""
        self.assertEqual(self.extractor.sampling_rate, 1000)
    
    def test_root_mean_square(self):
        """Test RMS calculation."""
        rms = self.extractor.root_mean_square(self.test_signal)
        self.assertIsInstance(rms, (float, np.floating))
        self.assertGreater(rms, 0)
    
    def test_mean_absolute_value(self):
        """Test MAV calculation."""
        mav = self.extractor.mean_absolute_value(self.test_signal)
        self.assertIsInstance(mav, (float, np.floating))
        self.assertGreater(mav, 0)
    
    def test_variance(self):
        """Test variance calculation."""
        var = self.extractor.variance(self.test_signal)
        self.assertIsInstance(var, (float, np.floating))
        self.assertGreaterEqual(var, 0)
    
    def test_waveform_length(self):
        """Test waveform length calculation."""
        wl = self.extractor.waveform_length(self.test_signal)
        self.assertIsInstance(wl, (float, np.floating))
        self.assertGreater(wl, 0)
    
    def test_zero_crossing_rate(self):
        """Test zero crossing rate calculation."""
        zcr = self.extractor.zero_crossing_rate(self.test_signal)
        self.assertIsInstance(zcr, (float, np.floating))
        self.assertGreaterEqual(zcr, 0)
        self.assertLessEqual(zcr, 1)
    
    def test_median_frequency(self):
        """Test median frequency calculation."""
        mf = self.extractor.median_frequency(self.test_signal)
        self.assertIsInstance(mf, (float, np.floating))
        self.assertGreater(mf, 0)
    
    def test_mean_frequency(self):
        """Test mean frequency calculation."""
        mf = self.extractor.mean_frequency(self.test_signal)
        self.assertIsInstance(mf, (float, np.floating))
        self.assertGreater(mf, 0)
    
    def test_spectral_entropy(self):
        """Test spectral entropy calculation."""
        se = self.extractor.spectral_entropy(self.test_signal)
        self.assertIsInstance(se, (float, np.floating))
        self.assertGreater(se, 0)
    
    def test_extract_all_features(self):
        """Test extraction of all features."""
        features = self.extractor.extract_all_features(self.test_signal)
        
        self.assertIsInstance(features, dict)
        expected_features = [
            'rms', 'mav', 'variance', 'waveform_length',
            'zero_crossing_rate', 'median_frequency',
            'mean_frequency', 'spectral_entropy'
        ]
        
        for feature_name in expected_features:
            self.assertIn(feature_name, features)
            self.assertIsInstance(features[feature_name], (float, np.floating))
    
    def test_extract_features_from_segments(self):
        """Test feature extraction from multiple segments."""
        segments = [
            self.test_signal[:500],
            self.test_signal[500:]
        ]
        
        features_list = self.extractor.extract_features_from_segments(segments)
        
        self.assertIsInstance(features_list, list)
        self.assertEqual(len(features_list), 2)
        self.assertIsInstance(features_list[0], dict)


if __name__ == '__main__':
    unittest.main()
