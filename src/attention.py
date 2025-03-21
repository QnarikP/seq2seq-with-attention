"""
Module: attention.py
Description: Implements a basic scaled dot-product attention mechanism.

This module contains a function to compute the attention weights
and the resulting weighted sum of values.
"""

import torch
import torch.nn.functional as F


def attention(query, key, value, mask=None, dropout=None):
    """
    ================================================================================
    Compute the Scaled Dot-Product Attention
    ================================================================================

    This function computes the attention output and attention weights given the
    query, key, and value tensors. It optionally applies a mask to prevent
    attention to certain positions and dropout to the attention weights.

    Parameters:
    ------------------------------------------------------------------------------
    query   : torch.Tensor
              Tensor of shape (batch_size, query_len, d_k) representing the query.
    key     : torch.Tensor
              Tensor of shape (batch_size, key_len, d_k) representing the keys.
    value   : torch.Tensor
              Tensor of shape (batch_size, key_len, d_v) representing the values.
    mask    : torch.Tensor, optional
              Mask tensor of shape (batch_size, query_len, key_len). Positions with
              a mask value of 0 will be ignored (set to a very negative number).
    dropout : torch.nn.Dropout, optional
              Dropout layer applied to the attention weights for regularization.

    Returns:
    ------------------------------------------------------------------------------
    output             : torch.Tensor
                         Tensor of shape (batch_size, query_len, d_v) with the weighted
                         sum of the values.
    attention_weights  : torch.Tensor
                         Tensor of shape (batch_size, query_len, key_len) representing
                         the attention probabilities.
    ================================================================================
    """
    # ---------------------------------------------------------
    # 1. Calculate the dot products between query and key.
    # ---------------------------------------------------------
    d_k = query.size(-1)  # Dimension of the key vectors.
    # Scale the dot products by square root of d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)

    # ---------------------------------------------------------
    # 2. Optionally apply a mask to prevent attention to some positions.
    # ---------------------------------------------------------
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # ---------------------------------------------------------
    # 3. Compute attention probabilities using the softmax function.
    # ---------------------------------------------------------
    attn = F.softmax(scores, dim=-1)

    # ---------------------------------------------------------
    # 4. Optionally apply dropout for regularization.
    # ---------------------------------------------------------
    if dropout is not None:
        attn = dropout(attn)

    # ---------------------------------------------------------
    # 5. Compute the weighted sum of the values using the attention weights.
    # ---------------------------------------------------------
    output = torch.matmul(attn, value)

    # Return both the output and the attention weights for analysis.
    return output, attn