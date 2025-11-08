"""
Neural Network Example

This module demonstrates how to use the VANTEX_AI neural network components
to build, train, and evaluate a simple feedforward neural network.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any

# Import our neural network components
from .layers import DenseLayer, Layer, DropoutLayer
from .trainer import Trainer, TrainingConfig, EarlyStopping
from .utils import (
    get_initializer, get_activation, one_hot_encode,
    train_test_split, normalize, accuracy
)

class SimpleNN(Layer):
    """A simple feedforward neural network with configurable architecture."""
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dims: List[int], 
                 output_dim: int,
                 activation: str = 'relu',
                 dropout: float = 0.0,
                 weight_init: str = 'he_normal'):
        """Initialize the neural network.
        
        Args:
            input_dim: Dimensionality of the input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimensionality of the output
            activation: Activation function to use in hidden layers
            dropout: Dropout rate (0.0 means no dropout)
            weight_init: Weight initialization method
        """
        super().__init__()
        self.layers = []
        self.dropout = dropout
        
        # Input layer
        prev_dim = input_dim
        
        # Hidden layers
        for i, dim in enumerate(hidden_dims):
            self.layers.append(DenseLayer(prev_dim, dim, activation=activation))
            if dropout > 0.0:
                self.layers.append(DropoutLayer(dropout))
            prev_dim = dim
        
        # Output layer (no activation, will be handled by loss function)
        self.layers.append(DenseLayer(prev_dim, output_dim))
        
        # Initialize parameters
        self._initialize_parameters(weight_init)
    
    def _initialize_parameters(self, init_method: str):
        """Initialize parameters using the specified method."""
        initializer = get_initializer(init_method)
        for layer in self.layers:
            if hasattr(layer, 'params'):
                for param_name, param in layer.params.items():
                    if 'W' in param_name:
                        # Use the specified initialization for weights
                        layer.params[param_name] = initializer(param.shape)
                    elif 'b' in param_name:
                        # Initialize biases to zero
                        layer.params[param_name] = np.zeros_like(param)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass through the network."""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get all parameters from all layers."""
        weights = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'get_weights'):
                layer_weights = layer.get_weights()
                for k, v in layer_weights.items():
                    weights[f'layer_{i}_{k}'] = v
        return weights
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set parameters for all layers."""
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'set_weights'):
                # Extract weights for this layer
                layer_weights = {}
                for k, v in weights.items():
                    if k.startswith(f'layer_{i}_'):
                        layer_weights[k.replace(f'layer_{i}_', '')] = v
                if layer_weights:
                    layer.set_weights(layer_weights)
    
    def train(self):
        """Set the network to training mode."""
        self.is_training = True
        for layer in self.layers:
            layer.train()
    
    def eval(self):
        """Set the network to evaluation mode."""
        self.is_training = False
        for layer in self.layers:
            layer.eval()


def generate_synthetic_data(n_samples: int = 1000, 
                          n_features: int = 20,
                          n_classes: int = 3,
                          random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic classification data.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X, y) where X is the feature matrix and y are the labels
    """
    np.random.seed(random_state)
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Create a non-linear decision boundary
    coef = np.random.randn(n_features, n_classes)
    logits = X @ coef + np.random.randn(n_samples, n_classes) * 0.1
    
    # Convert to probabilities and sample class labels
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    y = np.array([np.random.choice(n_classes, p=p) for p in probs])
    
    return X, y


def plot_training_history(history, metrics: List[str] = ['loss', 'accuracy']):
    """Plot training and validation metrics."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 4))
    
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Plot training metric
        train_metric = f'train_{metric}'
        if hasattr(history, train_metric):
            train_values = getattr(history, train_metric)
            ax.plot(train_values, label=f'Training {metric}')
        
        # Plot validation metric if available
        val_metric = f'val_{metric}'
        if hasattr(history, val_metric):
            val_values = getattr(history, val_metric)
            ax.plot(val_values, label=f'Validation {metric}')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} over Training')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    X, y = generate_synthetic_data(n_samples=2000, n_features=20, n_classes=3)
    
    # Convert labels to one-hot encoding
    y_one_hot = one_hot_encode(y)
    
    # Split into train and test sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_one_hot, test_size=0.2, random_state=42
    )
    
    # Normalize features
    X_mean, X_std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
    X_train = (X_train - X_mean) / (X_std + 1e-8)
    X_val = (X_val - X_mean) / (X_std + 1e-8)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Create model
    print("\nCreating model...")
    model = SimpleNN(
        input_dim=X_train.shape[1],
        hidden_dims=[64, 32],
        output_dim=y_one_hot.shape[1],
        activation='relu',
        dropout=0.2,
        weight_init='he_normal'
    )
    
    # Define training configuration
    config = TrainingConfig(
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        learning_rate_decay=0.95,
        decay_steps=100,
        early_stopping_patience=10,
        validation_split=0.1,
        shuffle=True,
        verbose=1,
        checkpoint_dir='checkpoints',
        checkpoint_freq=5
    )
    
    # Define loss function (cross-entropy)
    def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Compute cross-entropy
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return loss
    
    def cross_entropy_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute gradient of cross-entropy loss."""
        # For softmax output, the gradient is simply (y_pred - y_true)
        return (y_pred - y_true) / y_true.shape[0]
    
    # Define accuracy metric
    def accuracy_metric(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute accuracy."""
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_true, axis=1)
        return np.mean(y_pred_labels == y_true_labels)
    
    # Create trainer
    print("\nStarting training...")
    trainer = Trainer(model, config)
    trainer.compile(
        optimizer={
            'name': 'adam',
            'lr': config.learning_rate,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8
        },
        loss_fn=cross_entropy_loss,
        metrics=[accuracy_metric]
    )
    
    # Train the model
    history = trainer.fit(
        X_train, 
        y_train,
        x_val=X_val,
        y_val=y_val,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-4)
        ]
    )
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history, metrics=['loss', 'accuracy_metric'])
    
    # Evaluate on test set
    print("\nEvaluating on validation set...")
    test_loss, test_metrics = trainer.evaluate(X_val, y_val)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy_metric']:.4f}")
    
    # Example prediction
    print("\nExample prediction:")
    sample_idx = np.random.randint(0, len(X_val))
    sample = X_val[sample_idx:sample_idx+1]
    true_label = np.argmax(y_val[sample_idx])
    
    # Set model to evaluation mode
    model.eval()
    
    # Get prediction
    with np.printoptions(precision=3, suppress=True):
        logits = model.forward(sample)
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        pred_label = np.argmax(probs)
        
        print(f"Sample {sample_idx}:")
        print(f"  True class: {true_label}")
        print(f"  Predicted class: {pred_label}")
        print(f"  Class probabilities: {probs[0]}")


if __name__ == "__main__":
    main()
