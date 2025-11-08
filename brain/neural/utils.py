"""
Neural Network Utilities

This module contains utility functions and classes for neural networks.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
import json
import os
import gzip
import pickle
import hashlib
from pathlib import Path

# Type aliases
ArrayLike = Union[np.ndarray, List[float], List[List[float]]]
WeightInitializer = Callable[[Tuple[int, ...]], np.ndarray]


def get_initializer(name: str = 'he_normal', **kwargs) -> WeightInitializer:
    """Get a weight initializer function.
    
    Args:
        name: Name of the initializer. One of:
            - 'zeros': Initialize with zeros.
            - 'ones': Initialize with ones.
            - 'constant': Initialize with a constant value.
            - 'random_normal': Initialize with random values from a normal distribution.
            - 'random_uniform': Initialize with random values from a uniform distribution.
            - 'xavier': Xavier/Glorot initialization.
            - 'he_normal': He initialization with normal distribution.
            - 'he_uniform': He initialization with uniform distribution.
        **kwargs: Additional arguments for the initializer.
            - For 'constant': 'value' (default: 0.0)
            - For 'random_normal': 'mean' (default: 0.0), 'stddev' (default: 1.0)
            - For 'random_uniform': 'minval' (default: -1.0), 'maxval' (default: 1.0)
            - For 'xavier': 'scale' (default: 1.0)
            - For 'he_normal' and 'he_uniform': 'scale' (default: 2.0)
    
    Returns:
        A weight initializer function that takes a shape and returns an array.
    """
    if name == 'zeros':
        return lambda shape: np.zeros(shape)
    
    elif name == 'ones':
        return lambda shape: np.ones(shape)
    
    elif name == 'constant':
        value = kwargs.get('value', 0.0)
        return lambda shape: np.full(shape, value)
    
    elif name == 'random_normal':
        mean = kwargs.get('mean', 0.0)
        stddev = kwargs.get('stddev', 1.0)
        return lambda shape: np.random.normal(mean, stddev, shape)
    
    elif name == 'random_uniform':
        minval = kwargs.get('minval', -1.0)
        maxval = kwargs.get('maxval', 1.0)
        return lambda shape: np.random.uniform(minval, maxval, shape)
    
    elif name == 'xavier':
        scale = kwargs.get('scale', 1.0)
        def xavier(shape):
            if len(shape) < 2:
                raise ValueError("Xavier initializer requires at least 2D input shape")
            fan_in, fan_out = shape[0], np.prod(shape[1:])
            limit = np.sqrt(6.0 * scale / (fan_in + fan_out))
            return np.random.uniform(-limit, limit, shape)
        return xavier
    
    elif name == 'he_normal':
        scale = kwargs.get('scale', 2.0)
        def he_normal(shape):
            if len(shape) < 2:
                raise ValueError("He initializer requires at least 2D input shape")
            fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
            stddev = np.sqrt(scale / fan_in)
            return np.random.normal(0, stddev, shape)
        return he_normal
    
    elif name == 'he_uniform':
        scale = kwargs.get('scale', 2.0)
        def he_uniform(shape):
            if len(shape) < 2:
                raise ValueError("He initializer requires at least 2D input shape")
            fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
            limit = np.sqrt(6.0 * scale / fan_in)
            return np.random.uniform(-limit, limit, shape)
        return he_uniform
    
    else:
        raise ValueError(f"Unknown initializer: {name}")


def get_activation(name: str) -> Tuple[Callable, Callable]:
    """Get activation function and its derivative.
    
    Args:
        name: Name of the activation function. One of:
            - 'sigmoid': Sigmoid activation.
            - 'tanh': Hyperbolic tangent activation.
            - 'relu': Rectified Linear Unit.
            - 'leaky_relu': Leaky ReLU.
            - 'elu': Exponential Linear Unit.
            - 'selu': Scaled Exponential Linear Unit.
            - 'softmax': Softmax (returns (softmax, None) as derivative is handled differently).
            - 'linear': Linear activation (identity function).
    
    Returns:
        A tuple of (activation_function, derivative_function).
        For softmax, the second element is None.
    """
    if name == 'sigmoid':
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        def sigmoid_derivative(x):
            s = sigmoid(x)
            return s * (1 - s)
        
        return sigmoid, sigmoid_derivative
    
    elif name == 'tanh':
        def tanh_derivative(x):
            return 1 - np.tanh(x) ** 2
        
        return np.tanh, tanh_derivative
    
    elif name == 'relu':
        def relu(x):
            return np.maximum(0, x)
        
        def relu_derivative(x):
            return (x > 0).astype(float)
        
        return relu, relu_derivative
    
    elif name == 'leaky_relu':
        alpha = 0.01
        
        def leaky_relu(x):
            return np.where(x > 0, x, alpha * x)
        
        def leaky_relu_derivative(x):
            return np.where(x > 0, 1.0, alpha)
        
        return leaky_relu, leaky_relu_derivative
    
    elif name == 'elu':
        alpha = 1.0
        
        def elu(x):
            return np.where(x > 0, x, alpha * (np.exp(x) - 1))
        
        def elu_derivative(x):
            return np.where(x > 0, 1.0, alpha * np.exp(x))
        
        return elu, elu_derivative
    
    elif name == 'selu':
        # SELU parameters
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        
        def selu(x):
            return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
        
        def selu_derivative(x):
            return scale * np.where(x > 0, 1.0, alpha * np.exp(x))
        
        return selu, selu_derivative
    
    elif name == 'softmax':
        def softmax(x):
            # Numerically stable softmax
            e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return e_x / np.sum(e_x, axis=-1, keepdims=True)
        
        # Softmax derivative is handled differently in practice
        return softmax, None
    
    elif name == 'linear':
        def identity(x):
            return x
        
        def ones_like(x):
            return np.ones_like(x)
        
        return identity, ones_like
    
    else:
        raise ValueError(f"Unknown activation function: {name}")


def one_hot_encode(y: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """Convert class labels to one-hot encoded vectors.
    
    Args:
        y: Array of class labels (integers).
        num_classes: Number of classes. If None, it's inferred from the data.
    
    Returns:
        One-hot encoded array of shape (n_samples, num_classes).
    """
    if num_classes is None:
        num_classes = np.max(y) + 1
    
    if y.ndim > 1:
        y = y.ravel()
    
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                   random_state: Optional[int] = None) -> Tuple[np.ndarray, ...]:
    """Split arrays or matrices into random train and test subsets.
    
    Args:
        X: Input data.
        y: Target data.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Random seed for reproducibility.
    
    Returns:
        Tuple containing (X_train, X_test, y_train, y_test).
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def batch_generator(X: np.ndarray, y: np.ndarray, batch_size: int = 32, 
                  shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Generate batches of data.
    
    Args:
        X: Input data.
        y: Target data.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data before batching.
    
    Yields:
        Tuples of (X_batch, y_batch).
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]


def normalize(X: np.ndarray, axis: int = 0, epsilon: float = 1e-8) -> np.ndarray:
    """Normalize data to have zero mean and unit variance.
    
    Args:
        X: Input data.
        axis: Axis along which to normalize.
        epsilon: Small constant for numerical stability.
    
    Returns:
        Normalized data.
    """
    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)
    return (X - mean) / (std + epsilon)


def minmax_scale(X: np.ndarray, feature_range: Tuple[float, float] = (0, 1), 
                axis: int = 0) -> np.ndarray:
    """Scale features to a given range.
    
    Args:
        X: Input data.
        feature_range: Desired range of transformed data.
        axis: Axis along which to scale.
    
    Returns:
        Scaled data.
    """
    X_min = np.min(X, axis=axis, keepdims=True)
    X_range = np.ptp(X, axis=axis, keepdims=True)
    
    # Handle constant features
    X_range[X_range == 0] = 1
    
    # Scale to [0, 1] first
    X_std = (X - X_min) / X_range
    
    # Scale to desired range
    min_val, max_val = feature_range
    return X_std * (max_val - min_val) + min_val


def to_categorical(y: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """Convert class labels to one-hot encoded vectors (alias for one_hot_encode)."""
    return one_hot_encode(y, num_classes)


def save_weights(weights: Dict[str, np.ndarray], filepath: str):
    """Save model weights to a file.
    
    Args:
        weights: Dictionary of weight arrays.
        filepath: Path to save the weights.
    """
    if filepath.endswith('.npz'):
        np.savez_compressed(filepath, **weights)
    elif filepath.endswith('.h5') or filepath.endswith('.hdf5'):
        import h5py
        with h5py.File(filepath, 'w') as f:
            for name, value in weights.items():
                f.create_dataset(name, data=value)
    else:
        raise ValueError("Unsupported file format. Use .npz or .h5")


def load_weights(filepath: str) -> Dict[str, np.ndarray]:
    """Load model weights from a file.
    
    Args:
        filepath: Path to the weights file.
    
    Returns:
        Dictionary of weight arrays.
    """
    if filepath.endswith('.npz'):
        with np.load(filepath, allow_pickle=False) as data:
            return {name: data[name] for name in data.files}
    elif filepath.endswith('.h5') or filepath.endswith('.hdf5'):
        import h5py
        weights = {}
        with h5py.File(filepath, 'r') as f:
            for name in f.keys():
                weights[name] = f[name][()]
        return weights
    else:
        raise ValueError("Unsupported file format. Use .npz or .h5")


def get_file_hash(filepath: str, algorithm: str = 'md5', 
                chunk_size: int = 65536) -> str:
    """Calculate the hash of a file.
    
    Args:
        filepath: Path to the file.
        algorithm: Hash algorithm to use (e.g., 'md5', 'sha1', 'sha256').
        chunk_size: Size of chunks to read from the file.
    
    Returns:
        Hexadecimal digest of the file.
    """
    hash_func = hashlib.new(algorithm)
    
    with open(filepath, 'rb') as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            hash_func.update(data)
    
    return hash_func.hexdigest()


def ensure_dir(dirpath: Union[str, os.PathLike]):
    """Ensure that a directory exists, creating it if necessary."""
    os.makedirs(dirpath, exist_ok=True)


def save_pickle(obj: Any, filepath: str):
    """Save an object to a file using pickle with compression."""
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filepath: str) -> Any:
    """Load an object from a pickle file with compression support."""
    with gzip.open(filepath, 'rb') as f:
        return pickle.load(f)


def count_parameters(model: Any) -> int:
    """Count the number of trainable parameters in a model."""
    if hasattr(model, 'parameters'):
        # PyTorch-like model
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif hasattr(model, 'trainable_variables'):
        # TensorFlow/Keras-like model
        return sum(np.prod(v.shape) for v in model.trainable_variables)
    elif hasattr(model, 'get_weights'):
        # Our model interface
        return sum(np.prod(w.shape) for w in model.get_weights())
    else:
        raise ValueError("Unsupported model type")


def get_gpu_info() -> Dict[str, Any]:
    """Get information about available GPUs."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_info = {
                'gpu_count': gpu_count,
                'devices': []
            }
            
            for i in range(gpu_count):
                gpu_info['devices'].append({
                    'name': torch.cuda.get_device_name(i),
                    'capability': torch.cuda.get_device_capability(i),
                    'memory_allocated': torch.cuda.memory_allocated(i),
                    'memory_reserved': torch.cuda.memory_reserved(i),
                    'memory_cached': torch.cuda.memory_cached(i) if hasattr(torch.cuda, 'memory_cached') else None
                })
            
            return gpu_info
    except ImportError:
        pass
    
    return {'gpu_count': 0, 'devices': []}


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    import random
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set TensorFlow seed if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
