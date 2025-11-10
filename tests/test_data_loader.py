"""
Unit tests for the Zenodo data loader module.
"""

import unittest
import numpy as np
import os
import tempfile
import pandas as pd
from src.data_loader import ZenodoDataLoader, download_zenodo_dataset, load_zenodo_emg_data


class TestZenodoDataLoader(unittest.TestCase):
    """Test cases for ZenodoDataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = ZenodoDataLoader("14182446")
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test ZenodoDataLoader initialization."""
        self.assertEqual(self.loader.zenodo_record_id, "14182446")
        self.assertEqual(self.loader.base_url, "https://zenodo.org/records/14182446")
        self.assertEqual(self.loader.api_url, "https://zenodo.org/api/records/14182446")
        self.assertEqual(self.loader.download_dir, "data/zenodo")
    
    def test_initialization_custom_record(self):
        """Test initialization with custom record ID."""
        loader = ZenodoDataLoader("12345")
        self.assertEqual(loader.zenodo_record_id, "12345")
        self.assertEqual(loader.base_url, "https://zenodo.org/records/12345")
    
    def test_load_emg_csv_with_labels(self):
        """Test loading EMG data from CSV file with labels."""
        # Create a temporary CSV file
        csv_file = os.path.join(self.temp_dir, "test_data.csv")
        test_data = pd.DataFrame({
            'emg': np.random.randn(100),
            'fatigue_label': np.random.randint(0, 2, 100)
        })
        test_data.to_csv(csv_file, index=False)
        
        # Load the data
        signals, labels = self.loader.load_emg_csv(csv_file)
        
        self.assertEqual(len(signals), 100)
        self.assertEqual(len(labels), 100)
        self.assertIsInstance(signals, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)
    
    def test_load_emg_csv_without_labels(self):
        """Test loading EMG data from CSV file without labels."""
        # Create a temporary CSV file without labels
        csv_file = os.path.join(self.temp_dir, "test_data_no_labels.csv")
        test_data = pd.DataFrame({
            'emg': np.random.randn(100)
        })
        test_data.to_csv(csv_file, index=False)
        
        # Load the data
        signals, labels = self.loader.load_emg_csv(csv_file)
        
        self.assertEqual(len(signals), 100)
        self.assertIsNone(labels)
    
    def test_load_emg_csv_auto_detect_column(self):
        """Test auto-detection of signal column."""
        # Create a temporary CSV file with different column name
        csv_file = os.path.join(self.temp_dir, "test_data_auto.csv")
        test_data = pd.DataFrame({
            'signal_data': np.random.randn(100),
            'label': np.random.randint(0, 2, 100)
        })
        test_data.to_csv(csv_file, index=False)
        
        # Load the data (should auto-detect numeric column)
        signals, labels = self.loader.load_emg_csv(
            csv_file,
            signal_column='non_existent',
            label_column='label'
        )
        
        self.assertEqual(len(signals), 100)
        self.assertIsNotNone(signals)
    
    def test_load_emg_csv_invalid_file(self):
        """Test loading from non-existent file."""
        with self.assertRaises(ValueError):
            self.loader.load_emg_csv("non_existent_file.csv")
    
    def test_load_dataset_empty_directory(self):
        """Test loading dataset from empty directory."""
        # Create empty directory
        empty_dir = os.path.join(self.temp_dir, "empty")
        os.makedirs(empty_dir)
        
        # Load dataset
        dataset = self.loader.load_dataset(empty_dir)
        
        self.assertEqual(len(dataset), 0)
    
    def test_load_dataset_with_files(self):
        """Test loading dataset from directory with CSV files."""
        # Create test CSV files
        for i in range(3):
            csv_file = os.path.join(self.temp_dir, f"test_{i}.csv")
            test_data = pd.DataFrame({
                'emg': np.random.randn(50),
                'fatigue_label': np.random.randint(0, 2, 50)
            })
            test_data.to_csv(csv_file, index=False)
        
        # Load dataset
        dataset = self.loader.load_dataset(self.temp_dir)
        
        self.assertEqual(len(dataset), 3)
        for filename, (signals, labels) in dataset.items():
            self.assertEqual(len(signals), 50)
            self.assertEqual(len(labels), 50)
    
    def test_load_dataset_non_existent_directory(self):
        """Test loading dataset from non-existent directory."""
        with self.assertRaises(ValueError):
            self.loader.load_dataset("/non/existent/directory")
    
    def test_download_zenodo_dataset_function(self):
        """Test the download_zenodo_dataset convenience function."""
        # This will fail to download but should handle it gracefully
        result_dir = download_zenodo_dataset("14182446", self.temp_dir)
        self.assertEqual(result_dir, self.temp_dir)
    
    def test_load_zenodo_emg_data_function(self):
        """Test the load_zenodo_emg_data convenience function."""
        # Create test data
        os.makedirs(self.temp_dir, exist_ok=True)
        csv_file = os.path.join(self.temp_dir, "test.csv")
        test_data = pd.DataFrame({
            'emg': np.random.randn(50),
            'fatigue_label': np.random.randint(0, 2, 50)
        })
        test_data.to_csv(csv_file, index=False)
        
        # Load data
        dataset = load_zenodo_emg_data(self.temp_dir)
        self.assertIsInstance(dataset, dict)
        self.assertEqual(len(dataset), 1)


if __name__ == '__main__':
    unittest.main()
