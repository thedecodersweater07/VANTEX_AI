"""
Neural Network Trainer

This module contains classes and functions for training neural networks.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
import time
import json
import os

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    learning_rate_decay: float = 0.95
    decay_steps: int = 1000
    early_stopping_patience: int = 10
    validation_split: float = 0.1
    shuffle: bool = True
    verbose: int = 1
    checkpoint_dir: Optional[str] = None
    checkpoint_freq: int = 5  # Save checkpoint every N epochs
    use_amp: bool = False  # Automatic Mixed Precision
    clip_grad_norm: Optional[float] = 5.0  # Gradient clipping
    metrics: List[str] = field(default_factory=lambda: ['loss'])


@dataclass
class TrainingHistory:
    """Tracks training metrics over epochs."""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_metrics: Dict[str, List[float]] = field(default_factory=dict)
    val_metrics: Dict[str, List[float]] = field(default_factory=dict)
    learning_rates: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float('inf')
    
    def update(self, epoch: int, train_loss: float, val_loss: float, 
              train_metrics: Dict[str, float], val_metrics: Dict[str, float],
              learning_rate: float):
        """Update training history with new metrics."""
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.learning_rates.append(learning_rate)
        
        # Update metrics
        for name, value in train_metrics.items():
            if name not in self.train_metrics:
                self.train_metrics[name] = []
            self.train_metrics[name].append(value)
            
        for name, value in val_metrics.items():
            if name not in self.val_metrics:
                self.val_metrics[name] = []
            self.val_metrics[name].append(value)
        
        # Update best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert history to dictionary for serialization."""
        return {
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'learning_rates': self.learning_rates,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss
        }
    
    def save(self, filepath: str):
        """Save training history to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingHistory':
        """Load training history from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        history = cls()
        history.train_loss = data['train_loss']
        history.val_loss = data['val_loss']
        history.train_metrics = data['train_metrics']
        history.val_metrics = data['val_metrics']
        history.learning_rates = data['learning_rates']
        history.best_epoch = data['best_epoch']
        history.best_val_loss = data['best_val_loss']
        
        return history


class Trainer:
    """Handles training and evaluation of neural networks."""
    
    def __init__(self, model: Any, config: Optional[TrainingConfig] = None):
        """Initialize the trainer.
        
        Args:
            model: The neural network model to train.
            config: Training configuration. If None, default config is used.
        """
        self.model = model
        self.config = config or TrainingConfig()
        self.history = TrainingHistory()
        self.optimizer = None
        self.loss_fn = None
        self.metrics = {}
        self.global_step = 0
        self.epoch = 0
        
        # Create checkpoint directory if needed
        if self.config.checkpoint_dir:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
    
    def compile(self, optimizer: Any, loss_fn: Callable, metrics: Optional[List[Callable]] = None):
        """Configure the trainer.
        
        Args:
            optimizer: The optimizer to use (e.g., SGD, Adam).
            loss_fn: The loss function.
            metrics: List of metric functions to track during training.
        """
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
        # Initialize metrics
        self.metrics = {}
        if metrics:
            for metric in metrics:
                self.metrics[metric.__name__] = metric
    
    def train_step(self, x_batch: np.ndarray, y_batch: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Perform a single training step.
        
        Args:
            x_batch: Batch of input data.
            y_batch: Batch of target data.
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Forward pass
        y_pred = self.model.forward(x_batch)
        
        # Compute loss
        loss = self.loss_fn(y_pred, y_batch)
        
        # Backward pass
        grad = self.loss_fn.backward(y_pred, y_batch)
        self.model.backward(grad)
        
        # Update weights
        self.optimizer.step(self.model)
        
        # Compute metrics
        metrics = {}
        for name, metric_fn in self.metrics.items():
            metrics[name] = metric_fn(y_pred, y_batch)
        
        return loss, metrics
    
    def evaluate(self, x: np.ndarray, y: np.ndarray, batch_size: int = 32) -> Tuple[float, Dict[str, float]]:
        """Evaluate the model on the given data.
        
        Args:
            x: Input data.
            y: Target data.
            batch_size: Batch size for evaluation.
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        num_samples = x.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        total_loss = 0.0
        metrics_sum = {name: 0.0 for name in self.metrics}
        
        # Set model to evaluation mode
        self.model.eval()
        
        for i in range(num_batches):
            # Get batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            x_batch = x[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            
            # Forward pass
            y_pred = self.model.forward(x_batch)
            
            # Compute loss
            loss = self.loss_fn(y_pred, y_batch)
            total_loss += loss * (end_idx - start_idx)
            
            # Compute metrics
            for name, metric_fn in self.metrics.items():
                metrics_sum[name] += metric_fn(y_pred, y_batch) * (end_idx - start_idx)
        
        # Compute averages
        avg_loss = total_loss / num_samples
        avg_metrics = {name: value / num_samples for name, value in metrics_sum.items()}
        
        return avg_loss, avg_metrics
    
    def fit(self, x_train: np.ndarray, y_train: np.ndarray, 
            x_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            callbacks: Optional[List[Any]] = None) -> TrainingHistory:
        """Train the model.
        
        Args:
            x_train: Training input data.
            y_train: Training target data.
            x_val: Validation input data. If None, no validation is performed.
            y_val: Validation target data.
            callbacks: List of callback functions to call during training.
            
        Returns:
            Training history.
        """
        # Initialize callbacks
        callbacks = callbacks or []
        
        # Split training data if validation split is specified
        if x_val is None and y_val is None and self.config.validation_split > 0:
            from sklearn.model_selection import train_test_split
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train, 
                test_size=self.config.validation_split,
                random_state=42
            )
        
        # Initialize training
        num_samples = x_train.shape[0]
        num_batches = (num_samples + self.config.batch_size - 1) // self.config.batch_size
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.epoch, self.config.epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Shuffle training data
            if self.config.shuffle:
                indices = np.random.permutation(num_samples)
                x_train = x_train[indices]
                y_train = y_train[indices]
            
            # Set model to training mode
            self.model.train()
            
            # Initialize epoch metrics
            train_loss = 0.0
            train_metrics = {name: 0.0 for name in self.metrics}
            
            # Process batches
            for i in range(num_batches):
                # Get batch
                start_idx = i * self.config.batch_size
                end_idx = min((i + 1) * self.config.batch_size, num_samples)
                x_batch = x_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                
                # Training step
                batch_loss, batch_metrics = self.train_step(x_batch, y_batch)
                
                # Update metrics
                batch_size = end_idx - start_idx
                train_loss += batch_loss * batch_size
                for name in train_metrics:
                    train_metrics[name] += batch_metrics[name] * batch_size
                
                # Update learning rate if using decay
                if self.config.decay_steps > 0 and self.global_step % self.config.decay_steps == 0:
                    self.optimizer.lr *= self.config.learning_rate_decay
                
                self.global_step += 1
                
                # Call callbacks
                for callback in callbacks:
                    if hasattr(callback, 'on_batch_end'):
                        callback.on_batch_end(self.global_step, {'loss': batch_loss, **batch_metrics})
            
            # Compute average metrics for the epoch
            train_loss /= num_samples
            for name in train_metrics:
                train_metrics[name] /= num_samples
            
            # Evaluate on validation set
            val_loss = 0.0
            val_metrics = {name: 0.0 for name in self.metrics}
            
            if x_val is not None and y_val is not None:
                val_loss, val_metrics = self.evaluate(x_val, y_val)
                
                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    if self.config.checkpoint_dir:
                        self.save_checkpoint('best_model')
                else:
                    patience_counter += 1
                    
                    # Early stopping
                    if 0 < self.config.early_stopping_patience <= patience_counter:
                        if self.config.verbose > 0:
                            print(f'\nEarly stopping at epoch {epoch + 1}')
                        break
            
            # Update history
            self.history.update(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                learning_rate=self.optimizer.lr
            )
            
            # Print progress
            if self.config.verbose > 0:
                epoch_time = time.time() - epoch_start_time
                print(f'Epoch {epoch + 1}/{self.config.epochs} - {epoch_time:.1f}s - ', end='')
                print(f'loss: {train_loss:.4f} - ', end='')
                
                for name, value in train_metrics.items():
                    print(f'{name}: {value:.4f} - ', end='')
                
                if x_val is not None and y_val is not None:
                    print(f'val_loss: {val_loss:.4f} - ', end='')
                    for name, value in val_metrics.items():
                        print(f'val_{name}: {value:.4f} - ', end='')
                
                print()
            
            # Save checkpoint
            if (self.config.checkpoint_dir and 
                self.config.checkpoint_freq > 0 and 
                (epoch + 1) % self.config.checkpoint_freq == 0):
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}')
            
            # Call callbacks
            for callback in callbacks:
                if hasattr(callback, 'on_epoch_end'):
                    callback.on_epoch_end(epoch, {
                        'loss': train_loss,
                        'val_loss': val_loss,
                        **train_metrics,
                        **{f'val_{k}': v for k, v in val_metrics.items()}
                    })
        
        return self.history
    
    def predict(self, x: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Generate predictions for the input data.
        
        Args:
            x: Input data.
            batch_size: Batch size for prediction.
            
        Returns:
            Model predictions.
        """
        num_samples = x.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize predictions
        predictions = []
        
        for i in range(num_batches):
            # Get batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            x_batch = x[start_idx:end_idx]
            
            # Forward pass
            y_pred = self.model.forward(x_batch)
            predictions.append(y_pred)
        
        return np.concatenate(predictions, axis=0)
    
    def save_checkpoint(self, name: str):
        """Save a training checkpoint.
        
        Args:
            name: Name of the checkpoint.
        """
        if not self.config.checkpoint_dir:
            return
        
        checkpoint_dir = os.path.join(self.config.checkpoint_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model weights
        self.model.save_weights(os.path.join(checkpoint_dir, 'model_weights.npz'))
        
        # Save optimizer state
        if hasattr(self.optimizer, 'get_state'):
            optimizer_state = self.optimizer.get_state()
            np.savez_compressed(
                os.path.join(checkpoint_dir, 'optimizer.npz'), 
                **optimizer_state
            )
        
        # Save training state
        training_state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_val_loss': self.history.best_val_loss,
            'config': self.config.__dict__
        }
        
        with open(os.path.join(checkpoint_dir, 'training_state.json'), 'w') as f:
            json.dump(training_state, f, indent=2)
        
        # Save history
        self.history.save(os.path.join(checkpoint_dir, 'history.json'))
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load a training checkpoint.
        
        Args:
            checkpoint_dir: Directory containing the checkpoint.
        """
        # Load model weights
        self.model.load_weights(os.path.join(checkpoint_dir, 'model_weights.npz'))
        
        # Load optimizer state
        if hasattr(self.optimizer, 'set_state'):
            optimizer_state = np.load(os.path.join(checkpoint_dir, 'optimizer.npz'), allow_pickle=True)
            self.optimizer.set_state(dict(optimizer_state))
        
        # Load training state
        with open(os.path.join(checkpoint_dir, 'training_state.json'), 'r') as f:
            training_state = json.load(f)
        
        self.epoch = training_state['epoch']
        self.global_step = training_state['global_step']
        
        # Load history
        history_path = os.path.join(checkpoint_dir, 'history.json')
        if os.path.exists(history_path):
            self.history = TrainingHistory.load(history_path)
        
        return self


class LearningRateScheduler:
    """Learning rate scheduler with various scheduling strategies."""
    
    def __init__(self, initial_lr: float, schedule_type: str = 'constant', **kwargs):
        """Initialize the learning rate scheduler.
        
        Args:
            initial_lr: Initial learning rate.
            schedule_type: Type of learning rate schedule. One of:
                - 'constant': Constant learning rate.
                - 'step': Step decay.
                - 'exp': Exponential decay.
                - 'cosine': Cosine annealing.
                - 'cosine_restarts': Cosine annealing with restarts.
            **kwargs: Additional arguments for the specific schedule type.
        """
        self.initial_lr = initial_lr
        self.schedule_type = schedule_type.lower()
        self.kwargs = kwargs
        self.step_count = 0
        
        # Validate schedule type
        valid_types = ['constant', 'step', 'exp', 'cosine', 'cosine_restarts']
        if self.schedule_type not in valid_types:
            raise ValueError(f'Invalid schedule_type: {self.schedule_type}. Must be one of {valid_types}')
        
        # Initialize schedule-specific parameters
        if self.schedule_type == 'step':
            self.step_size = kwargs.get('step_size', 30)
            self.gamma = kwargs.get('gamma', 0.1)
        elif self.schedule_type == 'exp':
            self.gamma = kwargs.get('gamma', 0.95)
        elif self.schedule_type == 'cosine':
            self.T_max = kwargs.get('T_max', 100)
            self.eta_min = kwargs.get('eta_min', 0)
        elif self.schedule_type == 'cosine_restarts':
            self.T_0 = kwargs.get('T_0', 50)
            self.T_mult = kwargs.get('T_mult', 1)
            self.eta_min = kwargs.get('eta_min', 0)
    
    def step(self) -> float:
        """Update the learning rate and return the new value."""
        self.step_count += 1
        
        if self.schedule_type == 'constant':
            return self.initial_lr
        
        elif self.schedule_type == 'step':
            return self.initial_lr * (self.gamma ** (self.step_count // self.step_size))
        
        elif self.schedule_type == 'exp':
            return self.initial_lr * (self.gamma ** self.step_count)
        
        elif self.schedule_type == 'cosine':
            return self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * \
                   (1 + np.cos(np.pi * (self.step_count % self.T_max) / self.T_max))
        
        elif self.schedule_type == 'cosine_restarts':
            # Calculate which cycle we're in
            if self.T_mult == 1:
                T_cur = self.step_count % self.T_0
                T_i = self.T_0
            else:
                # Find the current cycle
                T_cur = self.step_count
                T_i = self.T_0
                while T_cur >= T_i:
                    T_cur -= T_i
                    T_i = int(self.T_mult * T_i)
                
                T_i = T_i // self.T_mult  # Previous cycle length
            
            return self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * \
                   (1 + np.cos(np.pi * T_cur / T_i))
        
        return self.initial_lr  # Fallback


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute accuracy for classification tasks."""
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        # Multi-class classification
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
    else:
        # Binary classification
        y_pred = (y_pred > 0.5).astype(int)
        y_true = (y_true > 0.5).astype(int)
    
    return np.mean(y_pred == y_true)


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute mean squared error."""
    return np.mean((y_pred - y_true) ** 2)


def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute mean absolute error."""
    return np.mean(np.abs(y_pred - y_true))


class EarlyStopping:
    """Early stopping callback to stop training when a monitored metric has stopped improving."""
    
    def __init__(self, monitor: str = 'val_loss', patience: int = 10, 
                 min_delta: float = 0.0, mode: str = 'min'):
        """Initialize the early stopping callback.
        
        Args:
            monitor: Metric to monitor (e.g., 'val_loss', 'val_accuracy').
            patience: Number of epochs with no improvement to wait before stopping.
            min_delta: Minimum change to qualify as an improvement.
            mode: One of 'min' or 'max'. In 'min' mode, training will stop when the
                quantity monitored has stopped decreasing; in 'max' mode it will stop
                when the quantity monitored has stopped increasing.
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else -float('inf')
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        
        if mode not in ['min', 'max']:
            raise ValueError(f'Invalid mode: {mode}. Must be one of ["min", "max"]')
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> bool:
        """Called at the end of each epoch.
        
        Args:
            epoch: Current epoch number.
            logs: Dictionary of metrics for the current epoch.
            
        Returns:
            bool: Whether to stop training (True) or continue (False).
        """
        current = logs.get(self.monitor)
        if current is None:
            return False
        
        if self.mode == 'min':
            if current < self.best_value - self.min_delta:
                self.best_value = current
                self.best_epoch = epoch
                self.wait = 0
            else:
                self.wait += 1
        else:  # mode == 'max'
            if current > self.best_value + self.min_delta:
                self.best_value = current
                self.best_epoch = epoch
                self.wait = 0
            else:
                self.wait += 1
        
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            return True
        
        return False
