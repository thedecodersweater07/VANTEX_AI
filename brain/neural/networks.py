"""
Neural Network Architectures

This module contains the core neural network architectures for VANTEX_AI.
"""

from typing import List, Dict, Any, Optional, Callable
import numpy as np
from abc import ABC, abstractmethod

class NeuralNetwork(ABC):
    """Base class for all neural network architectures."""
    
    def __init__(self):
        self.layers = []
        self.is_training = True
        
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        pass
    
    @abstractmethod
    def backward(self, grad: np.ndarray) -> None:
        """Backward pass through the network."""
        pass
    
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
    
    def add(self, layer):
        """Add a layer to the network."""
        self.layers.append(layer)
        return self
    
    def save_weights(self, filepath: str):
        """Save the model weights to a file."""
        weights = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'get_weights'):
                weights[f'layer_{i}'] = layer.get_weights()
        np.savez_compressed(filepath, **weights)
    
    def load_weights(self, filepath: str):
        """Load model weights from a file."""
        data = np.load(filepath, allow_pickle=True)
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'set_weights') and f'layer_{i}' in data:
                layer.set_weights(data[f'layer_{i}'])


class LSTMModel(NeuralNetwork):
    """LSTM-based neural network for sequence processing."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                 dropout: float = 0.2):
        super().__init__()
        from .layers import LSTMLayer, DenseLayer, DropoutLayer
        
        # Input layer
        self.add(LSTMLayer(input_dim, hidden_dims[0], return_sequences=len(hidden_dims) > 1))
        
        # Hidden layers
        for i in range(1, len(hidden_dims)):
            self.add(DropoutLayer(dropout))
            self.add(LSTMLayer(
                hidden_dims[i-1], 
                hidden_dims[i],
                return_sequences=i < len(hidden_dims) - 1
            ))
        
        # Output layer
        self.add(DenseLayer(hidden_dims[-1], output_dim, activation='softmax'))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad


class AttentionLayer:
    """Attention mechanism layer."""
    
    def __init__(self, hidden_dim: int):
        self.hidden_dim = hidden_dim
        self.W = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.V = np.random.randn(hidden_dim, 1) * 0.01
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for attention mechanism."""
        self.x = x
        
        # Calculate attention scores
        self.scores = np.tanh(np.dot(x, self.W))
        self.attention_weights = np.exp(np.dot(self.scores, self.V))
        self.attention_weights = self.attention_weights / np.sum(self.attention_weights, axis=1, keepdims=True)
        
        # Apply attention weights
        self.context = np.sum(self.attention_weights * x, axis=1)
        return self.context
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass for attention mechanism."""
        # Gradient w.r.t. attention weights
        d_attention_weights = np.expand_dims(grad, 1) * self.x
        
        # Gradient w.r.t. scores
        d_scores = np.dot(
            d_attention_weights * (self.attention_weights * (1 - self.attention_weights)),
            self.V.T
        )
        
        # Gradient w.r.t. W and V
        self.dW = np.dot(self.x.T, d_scores * (1 - np.tanh(np.dot(self.x, self.W)) ** 2))
        self.dV = np.dot(self.scores.T, d_attention_weights)
        
        # Gradient w.r.t. input
        dx = np.dot(d_scores, self.W.T)
        return dx
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        return {'W': self.W, 'V': self.V}
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        self.W = weights['W']
        self.V = weights['V']


class TransformerBlock:
    """Transformer block with multi-head self-attention and feed-forward layers."""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        
        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        
        # Feed-forward network
        self.ffn = [
            DenseLayer(embed_dim, ff_dim, activation='relu'),
            DenseLayer(ff_dim, embed_dim)
        ]
        
        # Layer normalization
        self.layernorm1 = LayerNormalization(embed_dim)
        self.layernorm2 = LayerNormalization(embed_dim)
        
        # Dropout
        self.dropout1 = DropoutLayer(dropout)
        self.dropout2 = DropoutLayer(dropout)
    
    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Forward pass for transformer block."""
        # Self-attention
        attn_output = self.attention(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed-forward network
        ffn_output = out1
        for layer in self.ffn:
            ffn_output = layer.forward(ffn_output)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass for transformer block."""
        # Backprop through layer norm 2 and add
        grad = self.layernorm2.backward(grad)
        
        # Backprop through FFN
        ffn_grad = grad
        for layer in reversed(self.ffn):
            ffn_grad = layer.backward(ffn_grad)
        
        # Backprop through layer norm 1 and add
        grad = self.layernorm1.backward(grad + ffn_grad)
        
        # Backprop through attention
        grad = self.attention.backward(grad)
        
        return grad


class MultiHeadAttention:
    """Multi-head self-attention mechanism."""
    
    def __init__(self, embed_dim: int, num_heads: int):
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear transformations for Q, K, V
        self.Wq = np.random.randn(embed_dim, embed_dim) * 0.01
        self.Wk = np.random.randn(embed_dim, embed_dim) * 0.01
        self.Wv = np.random.randn(embed_dim, embed_dim) * 0.01
        self.Wo = np.random.randn(embed_dim, embed_dim) * 0.01
        
        # Gradients
        self.dWq = np.zeros_like(self.Wq)
        self.dWk = np.zeros_like(self.Wk)
        self.dWv = np.zeros_like(self.Wv)
        self.dWo = np.zeros_like(self.Wo)
    
    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Forward pass for multi-head attention."""
        batch_size = Q.shape[0]
        
        # Linear transformations
        Q = np.dot(Q, self.Wq)  # (batch_size, seq_len, embed_dim)
        K = np.dot(K, self.Wk)  # (batch_size, seq_len, embed_dim)
        V = np.dot(V, self.Wv)  # (batch_size, seq_len, embed_dim)
        
        # Reshape to (batch_size, num_heads, seq_len, head_dim)
        Q = Q.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = self.softmax(scores, axis=-1)
        output = np.matmul(attention_weights, V)
        
        # Reshape back to (batch_size, seq_len, embed_dim)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.embed_dim)
        
        # Final linear layer
        output = np.dot(output, self.Wo)
        
        # Store for backward pass
        self.Q = Q
        self.K = K
        self.V = V
        self.attention_weights = attention_weights
        
        return output
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass for multi-head attention."""
        batch_size = grad.shape[0]
        
        # Gradient w.r.t. Wo
        self.dWo = np.dot(self.attention_output.transpose(0, 1, 3, 2), grad).sum(axis=0)
        
        # Gradient w.r.t. attention output
        grad = np.dot(grad, self.Wo.T)
        
        # Reshape grad to (batch_size, num_heads, seq_len, head_dim)
        grad = grad.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Gradients for V
        dV = np.matmul(self.attention_weights.transpose(0, 1, 3, 2), grad)
        self.dWv = np.dot(self.V_input.transpose(0, 2, 1), dV.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.embed_dim)).sum(axis=0)
        
        # Gradients for Q and K
        d_attention = np.matmul(grad, self.V.transpose(0, 1, 3, 2))
        d_scores = d_attention * self.attention_weights * (1 - self.attention_weights)
        
        dQ = np.matmul(d_scores, self.K) / np.sqrt(self.head_dim)
        dK = np.matmul(d_scores.transpose(0, 1, 3, 2), self.Q) / np.sqrt(self.head_dim)
        
        self.dWq = np.dot(self.Q_input.transpose(0, 2, 1), dQ.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.embed_dim)).sum(axis=0)
        self.dWk = np.dot(self.K_input.transpose(0, 2, 1), dK.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.embed_dim)).sum(axis=0)
        
        # Gradient w.r.t. input
        d_input = np.dot(dQ, self.Wq.T) + np.dot(dK, self.Wk.T) + np.dot(dV, self.Wv.T)
        
        return d_input
    
    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax.
        
        Args:
            x: Input array
            axis: Axis along which to compute the softmax
            
        Returns:
            Softmax output with the same shape as input
        """
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)
