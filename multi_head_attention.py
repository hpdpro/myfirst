"""
Multi-Head Attention Layer Implementation in Keras

This implementation follows the "Attention is All You Need" paper (Vaswani et al., 2017).
Multi-head attention allows the model to jointly attend to information from different
representation subspaces at different positions.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class MultiHeadAttention(layers.Layer):
    """
    Multi-Head Attention layer.

    Args:
        d_model: Dimensionality of the model (embedding dimension)
        num_heads: Number of attention heads
        dropout: Dropout rate (default: 0.1)

    Input shape:
        query: Tensor of shape (batch_size, seq_len_q, d_model)
        key: Tensor of shape (batch_size, seq_len_k, d_model)
        value: Tensor of shape (batch_size, seq_len_v, d_model)
        mask: Optional mask tensor

    Output shape:
        Tensor of shape (batch_size, seq_len_q, d_model)
    """

    def __init__(self, d_model, num_heads, dropout=0.1, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout

        # Dimension of each attention head
        self.depth = d_model // num_heads

        # Linear projections for Q, K, V
        self.wq = layers.Dense(d_model, name="query_projection")
        self.wk = layers.Dense(d_model, name="key_projection")
        self.wv = layers.Dense(d_model, name="value_projection")

        # Output projection
        self.dense = layers.Dense(d_model, name="output_projection")

        # Dropout layer
        self.dropout = layers.Dropout(dropout)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        Calculate the attention weights.

        Args:
            q: Query tensor of shape (..., seq_len_q, depth)
            k: Key tensor of shape (..., seq_len_k, depth)
            v: Value tensor of shape (..., seq_len_v, depth)
            mask: Optional mask tensor

        Returns:
            output: Attention output
            attention_weights: Attention weights
        """
        # Calculate attention scores: Q @ K^T / sqrt(d_k)
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        # Scale by square root of depth
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Add mask if provided
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # Softmax to get attention weights
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        # Apply dropout
        attention_weights = self.dropout(attention_weights)

        # Multiply by values
        output = tf.matmul(attention_weights, v)

        return output, attention_weights

    def call(self, query, key, value, mask=None, training=False):
        """
        Forward pass of multi-head attention.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional mask tensor
            training: Boolean indicating training mode

        Returns:
            output: Multi-head attention output
            attention_weights: Attention weights (for visualization)
        """
        batch_size = tf.shape(query)[0]

        # Linear projections
        q = self.wq(query)  # (batch_size, seq_len_q, d_model)
        k = self.wk(key)    # (batch_size, seq_len_k, d_model)
        v = self.wv(value)  # (batch_size, seq_len_v, d_model)

        # Split into multiple heads
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # Calculate attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask
        )
        # scaled_attention shape: (batch_size, num_heads, seq_len_q, depth)

        # Transpose and reshape
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                     (batch_size, -1, self.d_model))
        # (batch_size, seq_len_q, d_model)

        # Final linear projection
        output = self.dense(concat_attention)

        return output, attention_weights

    def get_config(self):
        """Return the config of the layer for serialization."""
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
        })
        return config


class TransformerBlock(layers.Layer):
    """
    Complete Transformer block with multi-head attention and feed-forward network.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dff: Dimension of feed-forward network
        dropout: Dropout rate
    """

    def __init__(self, d_model, num_heads, dff, dropout=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model),
        ])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x, mask=None, training=False):
        """Forward pass of transformer block."""
        # Multi-head attention with residual connection and layer norm
        attn_output, _ = self.attention(x, x, x, mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Feed-forward network with residual connection and layer norm
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


def create_padding_mask(seq):
    """
    Create padding mask for sequences.

    Args:
        seq: Sequence tensor of shape (batch_size, seq_len)

    Returns:
        Mask tensor of shape (batch_size, 1, 1, seq_len)
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """
    Create look-ahead mask for decoder self-attention.

    Args:
        size: Sequence length

    Returns:
        Mask tensor of shape (size, size)
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask
