"""
Example usage of Multi-Head Attention layer

This script demonstrates how to use the MultiHeadAttention layer
in different scenarios.
"""

import tensorflow as tf
import numpy as np
from multi_head_attention import (
    MultiHeadAttention,
    TransformerBlock,
    create_padding_mask,
    create_look_ahead_mask
)


def example_1_basic_usage():
    """Example 1: Basic usage of MultiHeadAttention"""
    print("=" * 60)
    print("Example 1: Basic Multi-Head Attention Usage")
    print("=" * 60)

    # Parameters
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8

    # Create random input
    query = tf.random.normal((batch_size, seq_len, d_model))
    key = tf.random.normal((batch_size, seq_len, d_model))
    value = tf.random.normal((batch_size, seq_len, d_model))

    # Create multi-head attention layer
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    # Forward pass
    output, attention_weights = mha(query, key, value)

    print(f"Input shape: {query.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Expected: (batch_size={batch_size}, num_heads={num_heads}, "
          f"seq_len_q={seq_len}, seq_len_k={seq_len})")
    print()


def example_2_self_attention():
    """Example 2: Self-attention (query, key, value are the same)"""
    print("=" * 60)
    print("Example 2: Self-Attention")
    print("=" * 60)

    batch_size = 1
    seq_len = 5
    d_model = 128
    num_heads = 4

    # Create input sequence
    x = tf.random.normal((batch_size, seq_len, d_model))

    # Create multi-head attention layer
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    # Self-attention: Q, K, V are all the same
    output, attention_weights = mha(x, x, x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nAttention weights (showing first head):")
    print(attention_weights[0, 0].numpy())
    print("\nNote: Each row sums to 1.0 (softmax normalization)")
    print(f"Row sums: {tf.reduce_sum(attention_weights[0, 0], axis=-1).numpy()}")
    print()


def example_3_with_mask():
    """Example 3: Using attention mask"""
    print("=" * 60)
    print("Example 3: Multi-Head Attention with Mask")
    print("=" * 60)

    batch_size = 1
    seq_len = 4
    d_model = 64
    num_heads = 2

    x = tf.random.normal((batch_size, seq_len, d_model))

    # Create look-ahead mask (for decoder)
    mask = create_look_ahead_mask(seq_len)
    print("Look-ahead mask (prevents attending to future positions):")
    print(mask.numpy())
    print()

    # Reshape mask for broadcasting
    mask = mask[tf.newaxis, tf.newaxis, :, :]

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    output, attention_weights = mha(x, x, x, mask=mask)

    print(f"Output shape: {output.shape}")
    print(f"\nAttention weights with mask (first head):")
    print(attention_weights[0, 0].numpy())
    print("\nNote: Upper triangle is zero (masked positions)")
    print()


def example_4_transformer_block():
    """Example 4: Complete Transformer block"""
    print("=" * 60)
    print("Example 4: Complete Transformer Block")
    print("=" * 60)

    batch_size = 2
    seq_len = 8
    d_model = 256
    num_heads = 8
    dff = 1024  # Feed-forward network dimension

    x = tf.random.normal((batch_size, seq_len, d_model))

    # Create transformer block
    transformer = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        dropout=0.1
    )

    # Forward pass
    output = transformer(x, training=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("\nTransformer block includes:")
    print("  1. Multi-head self-attention")
    print("  2. Add & Norm (residual connection + layer normalization)")
    print("  3. Feed-forward network")
    print("  4. Add & Norm")
    print()


def example_5_sequence_model():
    """Example 5: Building a simple sequence classification model"""
    print("=" * 60)
    print("Example 5: Sequence Classification Model")
    print("=" * 60)

    vocab_size = 10000
    max_len = 100
    d_model = 128
    num_heads = 4
    num_classes = 5

    # Build model
    inputs = tf.keras.Input(shape=(max_len,))

    # Embedding layer
    x = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)

    # Add positional encoding (simplified)
    positions = tf.range(start=0, limit=max_len, delta=1)
    position_embedding = tf.keras.layers.Embedding(max_len, d_model)(positions)
    x = x + position_embedding

    # Multi-head attention
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    attn_output, _ = mha(x, x, x)

    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(attn_output)

    # Classification head
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    print(model.summary())
    print()


def example_6_cross_attention():
    """Example 6: Cross-attention (encoder-decoder attention)"""
    print("=" * 60)
    print("Example 6: Cross-Attention")
    print("=" * 60)

    batch_size = 1
    encoder_seq_len = 10
    decoder_seq_len = 8
    d_model = 256
    num_heads = 8

    # Encoder output (context)
    encoder_output = tf.random.normal((batch_size, encoder_seq_len, d_model))

    # Decoder input (query)
    decoder_input = tf.random.normal((batch_size, decoder_seq_len, d_model))

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    # Cross-attention: Q from decoder, K and V from encoder
    output, attention_weights = mha(
        query=decoder_input,
        key=encoder_output,
        value=encoder_output
    )

    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Decoder input shape: {decoder_input.shape}")
    print(f"Cross-attention output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"\nNote: Attention weights shape is "
          f"(batch, num_heads, decoder_seq_len, encoder_seq_len)")
    print("Each decoder position attends to all encoder positions")
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("Multi-Head Attention Examples")
    print("=" * 60 + "\n")

    example_1_basic_usage()
    example_2_self_attention()
    example_3_with_mask()
    example_4_transformer_block()
    example_5_sequence_model()
    example_6_cross_attention()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    main()
