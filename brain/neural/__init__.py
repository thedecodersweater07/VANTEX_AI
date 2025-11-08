"""
Neural Network Module

This module contains the core neural network components for VANTEX_AI,
including various neural network architectures, layers, and utilities.
"""

from .layers import *
from .networks import *
from .trainer import *
from .utils import *

__all__ = [
    # Networks
    'NeuralNetwork',
    'LSTMModel',
    'AttentionLayer',
    'TransformerBlock',
    
    # Layers
    'DenseLayer',
    'LSTMLayer',
    'DropoutLayer',
    'BatchNormLayer',
    'Layer',
    'LayerNormalization',
    
    # Utils
    'TrainingConfig',
    'TrainingHistory',
    'Trainer',
    'LearningRateScheduler',
    'EarlyStopping',
    'accuracy',
    'mse',
    'mae'
]
