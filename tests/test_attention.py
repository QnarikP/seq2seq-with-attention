"""
Module: test_attention.py
Description: Unit tests for attention and self-attention modules.
Author: [Your Name]
Date: [Date]
"""

import torch
import pytest
from src.attention import attention
from src.self_attention import SelfAttention


def test_attention_output_shape():
    """
    Test the output shape of the attention function.
    """
    batch_size = 2
    query_len = 5
    key_len = 7
    d_k = 16
    d_v = 32

    # Create dummy tensors
    query = torch.randn(batch_size, query_len, d_k)
    key = torch.randn(batch_size, key_len, d_k)
    value = torch.randn(batch_size, key_len, d_v)

    # Compute attention output
    output, attn_weights = attention(query, key, value)

    # Check output shape: should be (batch_size, query_len, d_v)
    assert output.shape == (batch_size, query_len, d_v), f"Output shape mismatch: {output.shape}"

    # Check attention weights shape: should be (batch_size, query_len, key_len)
    assert attn_weights.shape == (
    batch_size, query_len, key_len), f"Attention weights shape mismatch: {attn_weights.shape}"


def test_self_attention_module():
    """
    Test the SelfAttention module with dummy input.
    """
    batch_size = 2
    seq_length = 10
    embed_dim = 64

    # Create dummy input tensor
    dummy_input = torch.randn(batch_size, seq_length, embed_dim)

    # Initialize SelfAttention module
    self_attn = SelfAttention(embed_dim)

    # Forward pass
    output, attn_weights = self_attn(dummy_input)

    # Check that output shape is the same as input shape
    assert output.shape == dummy_input.shape, f"SelfAttention output shape mismatch: {output.shape}"

    # Check attention weights shape: (batch_size, seq_length, seq_length)
    assert attn_weights.shape == (
    batch_size, seq_length, seq_length), f"Attention weights shape mismatch: {attn_weights.shape}"


if __name__ == "__main__":
    # Run tests manually if needed
    pytest.main([__file__])