"""
Neural Network Layers

This module contains various layer implementations for building neural networks.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List

class Layer:
    """Base class for all neural network layers."""
    
    def __init__(self):
        """Initialize the base layer with empty parameters and gradients."""
        self.params = {}
        self.grads = {}
        self.is_training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        raise NotImplementedError
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass."""
        raise NotImplementedError
    
    def train(self):
        """Set the layer to training mode."""
        self.is_training = True
    
    def eval(self):
        """Set the layer to evaluation mode."""
        self.is_training = False
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get the layer's weights."""
        return {}
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set the layer's weights."""
        pass


class DenseLayer(Layer):
    """Fully connected layer."""
    
    def __init__(self, input_dim: int, output_dim: int, activation: Optional[str] = None):
        """Initialize a dense layer.
        
        Args:
            input_dim: Dimensionality of the input features
            output_dim: Dimensionality of the output features
            activation: Activation function to use (e.g., 'relu', 'sigmoid', 'tanh', 'softmax')
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        
        # Initialize weights and biases
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.params['W'] = np.random.randn(input_dim, output_dim) * scale
        self.params['b'] = np.zeros((1, output_dim))
        
        # Initialize gradients
        self.grads = {
            'W': np.zeros_like(self.params['W']),
            'b': np.zeros_like(self.params['b'])
        }
        
        # Cache for backward pass
        self.cache = {}
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for dense layer."""
        self.cache['x'] = x
        
        # Linear transformation
        z = np.dot(x, self.params['W']) + self.params['b']
        
        # Apply activation if specified
        if self.activation == 'relu':
            return self.relu(z)
        elif self.activation == 'sigmoid':
            return self.sigmoid(z)
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'softmax':
            return self.softmax(z)
        else:
            return z
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass for dense layer."""
        x = self.cache['x']
        
        # Gradient of activation
        if hasattr(self, 'activation'):
            if self.activation == 'relu':
                grad = grad * self.relu_derivative()
            elif self.activation == 'sigmoid':
                grad = grad * self.sigmoid_derivative()
            elif self.activation == 'tanh':
                grad = grad * (1 - np.tanh(self.cache['z']) ** 2)
            elif self.activation == 'softmax':
                # Assumes grad is the derivative of the loss w.r.t. softmax output
                pass
        
        # Compute gradients
        m = x.shape[0]
        self.grads['W'] = np.dot(x.T, grad) / m
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True) / m
        
        # Compute gradient w.r.t. input
        dx = np.dot(grad, self.params['W'].T)
        
        return dx
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        self.cache['z'] = x
        return np.maximum(0, x)
    
    def relu_derivative(self) -> np.ndarray:
        """Derivative of ReLU."""
        return (self.cache['z'] > 0).astype(float)
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        self.cache['z'] = x
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self) -> np.ndarray:
        """Derivative of sigmoid."""
        s = self.sigmoid(self.cache['z'])
        return s * (1 - s)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation function."""
        # Numerically stable softmax
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        return {'W': self.params['W'], 'b': self.params['b']}
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        self.params['W'] = weights['W']
        self.params['b'] = weights['b']


class LSTMLayer(Layer):
    """Long Short-Term Memory (LSTM) layer."""
    
    def __init__(self, input_dim: int, hidden_dim: int, return_sequences: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.return_sequences = return_sequences
        
        # Input gate weights
        self.W_xi = self._init_weights((input_dim, hidden_dim))
        self.W_hi = self._init_weights((hidden_dim, hidden_dim))
        self.b_i = np.zeros((1, hidden_dim))
        
        # Forget gate weights
        self.W_xf = self._init_weights((input_dim, hidden_dim))
        self.W_hf = self._init_weights((hidden_dim, hidden_dim))
        self.b_f = np.ones((1, hidden_dim))  # Initialize forget gate bias to 1
        
        # Cell update weights
        self.W_xc = self._init_weights((input_dim, hidden_dim))
        self.W_hc = self._init_weights((hidden_dim, hidden_dim))
        self.b_c = np.zeros((1, hidden_dim))
        
        # Output gate weights
        self.W_xo = self._init_weights((input_dim, hidden_dim))
        self.W_ho = self._init_weights((hidden_dim, hidden_dim))
        self.b_o = np.zeros((1, hidden_dim))
        
        # Initialize gradients
        self.reset_gradients()
        
        # Cache for backward pass
        self.cache = {}
    
    def _init_weights(self, shape: Tuple[int, int]) -> np.ndarray:
        """Initialize weights using Xavier initialization."""
        return np.random.randn(*shape) * np.sqrt(2.0 / sum(shape))
    
    def reset_gradients(self):
        """Reset all gradients to zero."""
        self.dW_xi = np.zeros_like(self.W_xi)
        self.dW_hi = np.zeros_like(self.W_hi)
        self.db_i = np.zeros_like(self.b_i)
        
        self.dW_xf = np.zeros_like(self.W_xf)
        self.dW_hf = np.zeros_like(self.W_hf)
        self.db_f = np.zeros_like(self.b_f)
        
        self.dW_xc = np.zeros_like(self.W_xc)
        self.dW_hc = np.zeros_like(self.W_hc)
        self.db_c = np.zeros_like(self.b_c)
        
        self.dW_xo = np.zeros_like(self.W_xo)
        self.dW_ho = np.zeros_like(self.W_ho)
        self.db_o = np.zeros_like(self.b_o)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for LSTM."""
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden state and cell state
        h = np.zeros((batch_size, self.hidden_dim))
        c = np.zeros((batch_size, self.hidden_dim))
        
        # Store outputs if returning sequences
        if self.return_sequences:
            h_seq = np.zeros((batch_size, seq_len, self.hidden_dim))
        
        # Cache for backward pass
        self.cache = {'x': x, 'h': [], 'c': [], 'i': [], 'f': [], 'g': [], 'o': []}
        
        # Process each time step
        for t in range(seq_len):
            # Input at time step t
            x_t = x[:, t, :]
            
            # Input gate
            i = self.sigmoid(np.dot(x_t, self.W_xi) + np.dot(h, self.W_hi) + self.b_i)
            
            # Forget gate
            f = self.sigmoid(np.dot(x_t, self.W_xf) + np.dot(h, self.W_hf) + self.b_f)
            
            # Cell update
            g = np.tanh(np.dot(x_t, self.W_xc) + np.dot(h, self.W_hc) + self.b_c)
            
            # Update cell state
            c = f * c + i * g
            
            # Output gate
            o = self.sigmoid(np.dot(x_t, self.W_xo) + np.dot(h, self.W_ho) + self.b_o)
            
            # Update hidden state
            h = o * np.tanh(c)
            
            # Store for backward pass
            self.cache['h'].append(h)
            self.cache['c'].append(c)
            self.cache['i'].append(i)
            self.cache['f'].append(f)
            self.cache['g'].append(g)
            self.cache['o'].append(o)
            
            if self.return_sequences:
                h_seq[:, t, :] = h
        
        # Return either the final hidden state or all hidden states
        return h_seq if self.return_sequences else h
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass for LSTM."""
        x = self.cache['x']
        batch_size, seq_len, _ = x.shape
        
        # Initialize gradients
        dh_next = np.zeros((batch_size, self.hidden_dim))
        dc_next = np.zeros((batch_size, self.hidden_dim))
        dx = np.zeros_like(x)
        
        # Backpropagate through time
        for t in reversed(range(seq_len)):
            # Get cached values
            h_prev = self.cache['h'][t-1] if t > 0 else np.zeros((batch_size, self.hidden_dim))
            c_prev = self.cache['c'][t-1] if t > 0 else np.zeros((batch_size, self.hidden_dim))
            
            i = self.cache['i'][t]
            f = self.cache['f'][t]
            g = self.cache['g'][t]
            o = self.cache['o'][t]
            c = self.cache['c'][t]
            
            # Gradient of loss w.r.t. hidden state
            dh = grad[:, t, :] if self.return_sequences else (grad if t == seq_len - 1 else 0)
            dh = dh + dh_next
            
            # Gradient of loss w.r.t. output gate
            do = dh * np.tanh(c) * o * (1 - o)
            
            # Gradient of loss w.r.t. cell state
            dc = dc_next + dh * o * (1 - np.tanh(c) ** 2)
            
            # Gradient of loss w.r.t. input gate
            di = dc * g * i * (1 - i)
            
            # Gradient of loss w.r.t. forget gate
            df = dc * c_prev * f * (1 - f)
            
            # Gradient of loss w.r.t. cell update
            dg = dc * i * (1 - g ** 2)
            
            # Gradients of loss w.r.t. parameters
            x_t = x[:, t, :]
            
            # Input gate gradients
            self.dW_xi += np.dot(x_t.T, di) / batch_size
            self.dW_hi += np.dot(h_prev.T, di) / batch_size
            self.db_i += np.sum(di, axis=0, keepdims=True) / batch_size
            
            # Forget gate gradients
            self.dW_xf += np.dot(x_t.T, df) / batch_size
            self.dW_hf += np.dot(h_prev.T, df) / batch_size
            self.db_f += np.sum(df, axis=0, keepdims=True) / batch_size
            
            # Cell update gradients
            self.dW_xc += np.dot(x_t.T, dg) / batch_size
            self.dW_hc += np.dot(h_prev.T, dg) / batch_size
            self.db_c += np.sum(dg, axis=0, keepdims=True) / batch_size
            
            # Output gate gradients
            self.dW_xo += np.dot(x_t.T, do) / batch_size
            self.dW_ho += np.dot(h_prev.T, do) / batch_size
            self.db_o += np.sum(do, axis=0, keepdims=True) / batch_size
            
            # Gradients w.r.t. inputs
            dx_t = (np.dot(di, self.W_xi.T) + 
                   np.dot(df, self.W_xf.T) + 
                   np.dot(dg, self.W_xc.T) + 
                   np.dot(do, self.W_xo.T))
            
            # Gradients for next time step
            dh_next = (np.dot(di, self.W_hi.T) + 
                      np.dot(df, self.W_hf.T) + 
                      np.dot(dg, self.W_hc.T) + 
                      np.dot(do, self.W_ho.T))
            
            dc_next = f * dc
            
            # Store input gradient
            dx[:, t, :] = dx_t
        
        return dx
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        return {
            'W_xi': self.W_xi, 'W_hi': self.W_hi, 'b_i': self.b_i,
            'W_xf': self.W_xf, 'W_hf': self.W_hf, 'b_f': self.b_f,
            'W_xc': self.W_xc, 'W_hc': self.W_hc, 'b_c': self.b_c,
            'W_xo': self.W_xo, 'W_ho': self.W_ho, 'b_o': self.b_o
        }
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        self.W_xi = weights['W_xi']
        self.W_hi = weights['W_hi']
        self.b_i = weights['b_i']
        self.W_xf = weights['W_xf']
        self.W_hf = weights['W_hf']
        self.b_f = weights['b_f']
        self.W_xc = weights['W_xc']
        self.W_hc = weights['W_hc']
        self.b_c = weights['b_c']
        self.W_xo = weights['W_xo']
        self.W_ho = weights['W_ho']
        self.b_o = weights['b_o']


class DropoutLayer(Layer):
    """Dropout layer for regularization."""
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self.mask = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for dropout."""
        if self.is_training:
            # Create and apply mask during training
            self.mask = (np.random.random(x.shape) > self.p) / (1 - self.p)
            return x * self.mask
        else:
            # Scale activations during inference
            return x * (1 - self.p)
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass for dropout."""
        if self.is_training:
            return grad * self.mask
        else:
            return grad * (1 - self.p)


class BatchNormLayer(Layer):
    """Batch normalization layer."""
    
    def __init__(self, num_features: int, momentum: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        self.momentum = momentum
        self.eps = eps
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        self.cache = {}
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for batch normalization."""
        if self.is_training:
            # Calculate batch statistics
            mu = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            # Normalize
            x_hat = (x - mu) / np.sqrt(var + self.eps)
            
            # Store for backward pass
            self.cache = {'x': x, 'x_hat': x_hat, 'mu': mu, 'var': var}
        else:
            # Use running statistics during inference
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        
        # Scale and shift
        out = self.gamma * x_hat + self.beta
        return out
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass for batch normalization."""
        x, x_hat, mu, var = self.cache['x'], self.cache['x_hat'], self.cache['mu'], self.cache['var']
        m = x.shape[0]
        
        # Gradients of scale and shift parameters
        dgamma = np.sum(grad * x_hat, axis=0, keepdims=True)
        dbeta = np.sum(grad, axis=0, keepdims=True)
        
        # Gradient w.r.t. normalized input
        dx_hat = grad * self.gamma
        
        # Gradient w.r.t. input
        dvar = np.sum(dx_hat * (x - mu) * -0.5 * (var + self.eps) ** (-1.5), axis=0, keepdims=True)
        dmu = np.sum(dx_hat * -1 / np.sqrt(var + self.eps), axis=0, keepdims=True) + \
              dvar * np.sum(-2 * (x - mu), axis=0, keepdims=True) / m
        
        dx = dx_hat / np.sqrt(var + self.eps) + dvar * 2 * (x - mu) / m + dmu / m
        
        return dx
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        return {'gamma': self.gamma, 'beta': self.beta}
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        self.gamma = weights['gamma']
        self.beta = weights['beta']


class LayerNormalization(Layer):
    """Layer normalization layer."""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = np.ones((1, *normalized_shape) if isinstance(normalized_shape, tuple) else (1, normalized_shape))
        self.beta = np.zeros_like(self.gamma)
        self.eps = eps
        self.cache = {}
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for layer normalization."""
        # Calculate mean and variance along the last dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        
        # Store for backward pass
        self.cache = {'x': x, 'x_hat': x_hat, 'mean': mean, 'var': var}
        
        # Scale and shift
        out = self.gamma * x_hat + self.beta
        return out
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass for layer normalization."""
        x, x_hat, mean, var = self.cache['x'], self.cache['x_hat'], self.cache['mean'], self.cache['var']
        m = x.shape[-1]
        
        # Gradients of scale and shift parameters
        dgamma = np.sum(grad * x_hat, axis=0, keepdims=True)
        dbeta = np.sum(grad, axis=0, keepdims=True)
        
        # Gradient w.r.t. normalized input
        dx_hat = grad * self.gamma
        
        # Gradient w.r.t. input
        dvar = np.sum(dx_hat * (x - mean) * -0.5 * (var + self.eps) ** (-1.5), axis=-1, keepdims=True)
        dmean = np.sum(dx_hat * -1 / np.sqrt(var + self.eps), axis=-1, keepdims=True) + \
                dvar * np.sum(-2 * (x - mean), axis=-1, keepdims=True) / m
        
        dx = dx_hat / np.sqrt(var + self.eps) + dvar * 2 * (x - mean) / m + dmean / m
        
        return dx
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        return {'gamma': self.gamma, 'beta': self.beta}
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        self.gamma = weights['gamma']
        self.beta = weights['beta']
