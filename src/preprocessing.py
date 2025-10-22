"""
Preprocessing module for EMG data.
Handles data loading, cleaning, and preparation for feature extraction.
"""

import numpy as np
import pandas as pd
from scipy import signal


class EMGPreprocessor:
    """
    Preprocessor for EMG (Electromyography) data.
    """
    
    def __init__(self, sampling_rate=1000):
        """
        Initialize the EMG preprocessor.
        
        Args:
            sampling_rate (int): Sampling rate of the EMG signal in Hz
        """
        self.sampling_rate = sampling_rate
    
    def load_data(self, filepath):
        """
        Load EMG data from a CSV file.
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded EMG data
        """
        try:
            data = pd.read_csv(filepath)
            return data
        except Exception as e:
            raise ValueError(f"Error loading data from {filepath}: {str(e)}")
    
    def bandpass_filter(self, data, lowcut=20, highcut=450, order=4):
        """
        Apply a bandpass filter to remove noise from EMG signals.
        
        Args:
            data (np.ndarray): Raw EMG signal
            lowcut (float): Lower cutoff frequency in Hz
            highcut (float): Upper cutoff frequency in Hz
            order (int): Filter order
            
        Returns:
            np.ndarray: Filtered EMG signal
        """
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = signal.butter(order, [low, high], btype='band')
        filtered_signal = signal.filtfilt(b, a, data)
        
        return filtered_signal
    
    def normalize(self, data):
        """
        Normalize the EMG signal to zero mean and unit variance.
        
        Args:
            data (np.ndarray): EMG signal
            
        Returns:
            np.ndarray: Normalized EMG signal
        """
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return data - mean
        
        return (data - mean) / std
    
    def segment_signal(self, signal_data, window_size, overlap=0.5):
        """
        Segment the EMG signal into windows for analysis.
        
        Args:
            signal_data (np.ndarray): EMG signal
            window_size (int): Size of each window in samples
            overlap (float): Overlap between windows (0 to 1)
            
        Returns:
            list: List of signal segments
        """
        step_size = int(window_size * (1 - overlap))
        segments = []
        
        for start in range(0, len(signal_data) - window_size + 1, step_size):
            segment = signal_data[start:start + window_size]
            segments.append(segment)
        
        return segments
    
    def preprocess_pipeline(self, signal_data, apply_filter=True, apply_normalization=True):
        """
        Complete preprocessing pipeline for EMG data.
        
        Args:
            signal_data (np.ndarray): Raw EMG signal
            apply_filter (bool): Whether to apply bandpass filter
            apply_normalization (bool): Whether to normalize the signal
            
        Returns:
            np.ndarray: Preprocessed EMG signal
        """
        processed_signal = signal_data.copy()
        
        if apply_filter:
            processed_signal = self.bandpass_filter(processed_signal)
        
        if apply_normalization:
            processed_signal = self.normalize(processed_signal)
        
        return processed_signal
