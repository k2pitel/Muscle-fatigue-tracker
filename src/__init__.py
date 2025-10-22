"""
Muscle Fatigue Tracker - ML-based EMG fatigue detection system.
"""

from .preprocessing import EMGPreprocessor
from .feature_extraction import EMGFeatureExtractor
from .models import FatigueClassifier
from .pipeline import FatigueDetectionPipeline, train_multiple_models
from .visualization import (
    plot_emg_signal,
    plot_comparison_signals,
    plot_feature_distributions,
    plot_confusion_matrix,
    plot_model_comparison
)

__all__ = [
    'EMGPreprocessor',
    'EMGFeatureExtractor',
    'FatigueClassifier',
    'FatigueDetectionPipeline',
    'train_multiple_models',
    'plot_emg_signal',
    'plot_comparison_signals',
    'plot_feature_distributions',
    'plot_confusion_matrix',
    'plot_model_comparison'
]
