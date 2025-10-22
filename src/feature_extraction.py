"""
Feature extraction module for EMG signals.
Extracts time-domain and frequency-domain features for muscle fatigue detection.
"""

import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq


class EMGFeatureExtractor:
    """
    Feature extractor for EMG signals.
    """
    
    def __init__(self, sampling_rate=1000):
        """
        Initialize the feature extractor.
        
        Args:
            sampling_rate (int): Sampling rate of the EMG signal in Hz
        """
        self.sampling_rate = sampling_rate
    
    def root_mean_square(self, signal_data):
        """
        Calculate Root Mean Square (RMS) of the signal.
        RMS is a key indicator of muscle activation level.
        
        Args:
            signal_data (np.ndarray): EMG signal
            
        Returns:
            float: RMS value
        """
        return np.sqrt(np.mean(signal_data ** 2))
    
    def mean_absolute_value(self, signal_data):
        """
        Calculate Mean Absolute Value (MAV) of the signal.
        
        Args:
            signal_data (np.ndarray): EMG signal
            
        Returns:
            float: MAV value
        """
        return np.mean(np.abs(signal_data))
    
    def variance(self, signal_data):
        """
        Calculate variance of the signal.
        
        Args:
            signal_data (np.ndarray): EMG signal
            
        Returns:
            float: Variance value
        """
        return np.var(signal_data)
    
    def waveform_length(self, signal_data):
        """
        Calculate waveform length (cumulative length of the waveform).
        
        Args:
            signal_data (np.ndarray): EMG signal
            
        Returns:
            float: Waveform length
        """
        return np.sum(np.abs(np.diff(signal_data)))
    
    def zero_crossing_rate(self, signal_data, threshold=0):
        """
        Calculate zero crossing rate.
        
        Args:
            signal_data (np.ndarray): EMG signal
            threshold (float): Threshold for zero crossing
            
        Returns:
            float: Zero crossing rate
        """
        sign_changes = np.diff(np.sign(signal_data))
        zero_crossings = np.sum(np.abs(sign_changes) > threshold)
        return zero_crossings / len(signal_data)
    
    def median_frequency(self, signal_data):
        """
        Calculate median frequency of the power spectrum.
        Median frequency decreases with muscle fatigue.
        
        Args:
            signal_data (np.ndarray): EMG signal
            
        Returns:
            float: Median frequency in Hz
        """
        # Compute FFT
        N = len(signal_data)
        yf = fft(signal_data)
        xf = fftfreq(N, 1 / self.sampling_rate)
        
        # Get positive frequencies only
        positive_freqs = xf[:N//2]
        power_spectrum = np.abs(yf[:N//2]) ** 2
        
        # Calculate median frequency
        cumsum = np.cumsum(power_spectrum)
        median_power = cumsum[-1] / 2
        median_freq_idx = np.where(cumsum >= median_power)[0][0]
        
        return positive_freqs[median_freq_idx]
    
    def mean_frequency(self, signal_data):
        """
        Calculate mean frequency of the power spectrum.
        Mean frequency also decreases with muscle fatigue.
        
        Args:
            signal_data (np.ndarray): EMG signal
            
        Returns:
            float: Mean frequency in Hz
        """
        N = len(signal_data)
        yf = fft(signal_data)
        xf = fftfreq(N, 1 / self.sampling_rate)
        
        # Get positive frequencies only
        positive_freqs = xf[:N//2]
        power_spectrum = np.abs(yf[:N//2]) ** 2
        
        # Calculate mean frequency
        mean_freq = np.sum(positive_freqs * power_spectrum) / np.sum(power_spectrum)
        
        return mean_freq
    
    def spectral_entropy(self, signal_data):
        """
        Calculate spectral entropy of the signal.
        
        Args:
            signal_data (np.ndarray): EMG signal
            
        Returns:
            float: Spectral entropy
        """
        N = len(signal_data)
        yf = fft(signal_data)
        power_spectrum = np.abs(yf[:N//2]) ** 2
        
        # Normalize to get probability distribution
        power_spectrum = power_spectrum / np.sum(power_spectrum)
        
        # Calculate entropy
        entropy = -np.sum(power_spectrum * np.log2(power_spectrum + 1e-12))
        
        return entropy
    
    def extract_all_features(self, signal_data):
        """
        Extract all features from the signal.
        
        Args:
            signal_data (np.ndarray): EMG signal
            
        Returns:
            dict: Dictionary containing all extracted features
        """
        features = {
            'rms': self.root_mean_square(signal_data),
            'mav': self.mean_absolute_value(signal_data),
            'variance': self.variance(signal_data),
            'waveform_length': self.waveform_length(signal_data),
            'zero_crossing_rate': self.zero_crossing_rate(signal_data),
            'median_frequency': self.median_frequency(signal_data),
            'mean_frequency': self.mean_frequency(signal_data),
            'spectral_entropy': self.spectral_entropy(signal_data)
        }
        
        return features
    
    def extract_features_from_segments(self, segments):
        """
        Extract features from multiple signal segments.
        
        Args:
            segments (list): List of signal segments
            
        Returns:
            list: List of feature dictionaries
        """
        all_features = []
        
        for segment in segments:
            features = self.extract_all_features(segment)
            all_features.append(features)
        
        return all_features
