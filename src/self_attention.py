"""
Module: self_attention.py
Description: Implements a self-attention mechanism using the basic attention function.
Author: [Your Name]
Date: [Date]

This module defines a PyTorch module for self-attention. It projects an input
sequence into query, key, and value spaces and then computes self-attention.
"""

import torch
import torch.nn as nn
from .attention import attention  # Import the attention function from attention.py


class SelfAttention(nn.Module):
    """
    ================================================================================
    Self-Attention Module
    ================================================================================

    This module computes self-attention for a given input tensor. It projects the input
    into query, key, and value tensors using linear layers, applies the attention function,
    and then projects the result back to the original embedding space.

    Parameters:
    ------------------------------------------------------------------------------
    embed_dim : int
                Dimension of the input embeddings.
    dropout   : float, optional (default=0.1)
                Dropout probability applied to the attention weights.
    ================================================================================
    """

    def __init__(self, embed_dim, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim

        # Linear layers for projecting the input to queries, keys, and values.
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        # Output linear layer to combine the attention output.
        self.out_linear = nn.Linear(embed_dim, embed_dim)

        # Dropout layer to be applied to the attention weights.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        ================================================================================
        Forward Pass for Self-Attention
        ================================================================================

        Parameters:
        ------------------------------------------------------------------------------
        x    : torch.Tensor
               Input tensor of shape (batch_size, sequence_length, embed_dim).
        mask : torch.Tensor, optional
               Mask tensor of shape (batch_size, sequence_length, sequence_length) to
               prevent attention to specific positions.

        Returns:
        ------------------------------------------------------------------------------
        output : torch.Tensor
                 Tensor of shape (batch_size, sequence_length, embed_dim) after applying
                 self-attention.
        attn   : torch.Tensor
                 Attention weights of shape (batch_size, sequence_length, sequence_length).
        ================================================================================
        """
        # ---------------------------------------------------------
        # 1. Linear projection of input to queries, keys, and values.
        # ---------------------------------------------------------
        query = self.q_linear(x)
        key = self.k_linear(x)
        value = self.v_linear(x)

        # ---------------------------------------------------------
        # 2. Compute the attention output and weights using the attention function.
        # ---------------------------------------------------------
        attn_output, attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # ---------------------------------------------------------
        # 3. Apply a final linear projection to the attention output.
        # ---------------------------------------------------------
        output = self.out_linear(attn_output)

        return output, attn