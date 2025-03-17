"""
Module: seq2seq.py
Description: Implements a Sequence-to-Sequence (Seq2Seq) model with attention
             for English-to-Russian translation. The model consists of an
             encoder, a decoder that integrates an attention mechanism, and
             a wrapper class for the complete seq2seq architecture.
Author: [Your Name]
Date: [Date]

This file contains:
    - Encoder: Encodes the source sentence.
    - Decoder: Generates the target sentence while attending to encoder outputs.
    - Seq2Seq: Wrapper that runs the encoder and decoder together.

A basic test block is provided at the end for verifying input/output shapes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the attention function defined in src/attention.py
from .attention import attention


# ================================================================================
#                           Encoder Module
# ================================================================================
class Encoder(nn.Module):
    """
    ================================================================================
    Encoder Module
    ================================================================================

    Encodes an input sentence into hidden representations using an embedding
    layer followed by an LSTM.

    Parameters:
    ------------------------------------------------------------------------------
    input_dim  : int
                 Size of the source vocabulary.
    embed_dim  : int
                 Dimension of the embedding vectors.
    hidden_dim : int
                 Dimension of the LSTM hidden states.
    num_layers : int, optional (default=1)
                 Number of layers in the LSTM.
    dropout    : float, optional (default=0.1)
                 Dropout probability for the LSTM.
    ================================================================================
    """

    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers=1, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            dropout=dropout, batch_first=True)

    def forward(self, src):
        """
        ------------------------------------------------------------------------------
        Forward Pass for the Encoder.

        Parameters:
        ------------------------------------------------------------------------------
        src : torch.Tensor
              Input tensor of shape (batch_size, src_len) containing token indices.

        Returns:
        ------------------------------------------------------------------------------
        outputs : torch.Tensor
                  LSTM outputs for each time step, shape (batch_size, src_len, hidden_dim).
        hidden  : torch.Tensor
                  Final hidden states from the LSTM.
        cell    : torch.Tensor
                  Final cell states from the LSTM.
        ------------------------------------------------------------------------------
        """
        # ------------------------------
        # 1. Embedding the source sentence.
        # ------------------------------
        embedded = self.embedding(src)  # shape: [batch_size, src_len, embed_dim]

        # ------------------------------
        # 2. Passing through the LSTM.
        # ------------------------------
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs: [batch_size, src_len, hidden_dim]

        return outputs, hidden, cell


# ================================================================================
#                           Decoder Module
# ================================================================================
class Decoder(nn.Module):
    """
    ================================================================================
    Decoder Module with Attention
    ================================================================================

    Decodes the encoder's output into the target sentence by applying an attention
    mechanism at each time step.

    Parameters:
    ------------------------------------------------------------------------------
    output_dim : int
                 Size of the target vocabulary.
    embed_dim  : int
                 Dimension of the embedding vectors.
    hidden_dim : int
                 Dimension of the LSTM hidden states.
    num_layers : int, optional (default=1)
                 Number of layers in the LSTM.
    dropout    : float, optional (default=0.1)
                 Dropout probability for the embedding layer.
    ================================================================================
    """

    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim)
        # The LSTM takes the concatenated embedding and context vector as input.
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, num_layers,
                            dropout=dropout, batch_first=True)
        # Final prediction layer: concatenates LSTM output, context vector, and embedding.
        self.fc_out = nn.Linear(hidden_dim * 2 + embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        """
        ------------------------------------------------------------------------------
        Forward Pass for the Decoder.

        Parameters:
        ------------------------------------------------------------------------------
        input            : torch.Tensor
                           Current input token indices of shape (batch_size).
        hidden, cell     : torch.Tensor
                           Hidden and cell states from the previous time step.
        encoder_outputs  : torch.Tensor
                           Encoder outputs of shape (batch_size, src_len, hidden_dim).

        Returns:
        ------------------------------------------------------------------------------
        prediction   : torch.Tensor
                       Output logits for the current time step, shape (batch_size, output_dim).
        hidden, cell : torch.Tensor
                       Updated hidden and cell states.
        attn_weights : torch.Tensor
                       Attention weights of shape (batch_size, 1, src_len).
        ------------------------------------------------------------------------------
        """
        # ------------------------------
        # 1. Embed the current input token.
        # ------------------------------
        input = input.unsqueeze(1)  # shape: [batch_size, 1]
        embedded = self.dropout(self.embedding(input))  # shape: [batch_size, 1, embed_dim]

        # ------------------------------
        # 2. Compute Attention Context.
        # ------------------------------
        # Use the last layer's hidden state as the query.
        query = hidden[-1].unsqueeze(1)  # shape: [batch_size, 1, hidden_dim]
        # Use encoder_outputs as both keys and values.
        context, attn_weights = attention(query, encoder_outputs, encoder_outputs)
        # context: [batch_size, 1, hidden_dim]

        # ------------------------------
        # 3. Concatenate the embedded token and context vector.
        # ------------------------------
        lstm_input = torch.cat((embedded, context), dim=2)  # shape: [batch_size, 1, embed_dim+hidden_dim]

        # ------------------------------
        # 4. Pass through the LSTM.
        # ------------------------------
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # output: [batch_size, 1, hidden_dim]

        # ------------------------------
        # 5. Generate Prediction.
        # ------------------------------
        # Squeeze out the sequence length dimension.
        output = output.squeeze(1)  # [batch_size, hidden_dim]
        context = context.squeeze(1)  # [batch_size, hidden_dim]
        embedded = embedded.squeeze(1)  # [batch_size, embed_dim]

        # Concatenate output, context, and embedded token to form final prediction input.
        prediction_input = torch.cat((output, context, embedded), dim=1)
        prediction = self.fc_out(prediction_input)  # [batch_size, output_dim]

        return prediction, hidden, cell, attn_weights


# ================================================================================
#                           Seq2Seq Model Wrapper
# ================================================================================
class Seq2Seq(nn.Module):
    """
    ================================================================================
    Seq2Seq Model with Attention
    ================================================================================

    Wraps the encoder and decoder into a single module for end-to-end training.

    Parameters:
    ------------------------------------------------------------------------------
    encoder : Encoder
              An instance of the Encoder module.
    decoder : Decoder
              An instance of the Decoder module.
    device  : torch.device
              Device to run the model (CPU or GPU).
    ================================================================================
    """

    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        ------------------------------------------------------------------------------
        Forward Pass for the Seq2Seq Model.

        Parameters:
        ------------------------------------------------------------------------------
        src                  : torch.Tensor
                               Source sentence tensor of shape (batch_size, src_len).
        trg                  : torch.Tensor
                               Target sentence tensor of shape (batch_size, trg_len).
        teacher_forcing_ratio: float, optional (default=0.5)
                               Probability of using teacher forcing.

        Returns:
        ------------------------------------------------------------------------------
        outputs    : torch.Tensor
                     Logits for each token in the target sequence,
                     shape (batch_size, trg_len, output_dim).
        attentions : torch.Tensor
                     Attention weights for each decoding step,
                     shape (batch_size, trg_len, src_len).
        ------------------------------------------------------------------------------
        """
        batch_size = src.size(0)
        trg_len = trg.size(1)
        output_dim = self.decoder.fc_out.out_features

        # Initialize tensor to store decoder outputs.
        outputs = torch.zeros(batch_size, trg_len, output_dim).to(self.device)
        # Initialize tensor to store attention weights.
        attentions = torch.zeros(batch_size, trg_len, src.size(1)).to(self.device)

        # ------------------------------
        # 1. Encode the source sentence.
        # ------------------------------
        encoder_outputs, hidden, cell = self.encoder(src)

        # ------------------------------
        # 2. Initialize the decoder input with <sos> token.
        #    Assumes trg[:, 0] is the start-of-sequence token.
        # ------------------------------
        input = trg[:, 0]

        # ------------------------------
        # 3. Decode each time step.
        # ------------------------------
        for t in range(1, trg_len):
            prediction, hidden, cell, attn_weights = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = prediction
            # attn_weights shape is [batch_size, 1, src_len] -> squeeze the sequence dim.
            attentions[:, t, :] = attn_weights.squeeze(1)

            # Determine next input: teacher forcing vs. model prediction.
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = prediction.argmax(1)  # [batch_size]
            input = trg[:, t] if teacher_force else top1

        return outputs, attentions


# ================================================================================
#                           Main Test Block
# ================================================================================
if __name__ == '__main__':
    # This block runs a simple forward pass test with dummy data.

    # ------------------------------
    # 1. Define the device.
    # ------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------------
    # 2. Set hyperparameters and dummy data dimensions.
    # ------------------------------
    INPUT_DIM = 1000  # Source vocabulary size (e.g., English)
    OUTPUT_DIM = 1000  # Target vocabulary size (e.g., Russian)
    EMBED_DIM = 256  # Embedding dimension for both encoder and decoder
    HIDDEN_DIM = 512  # Hidden state dimension for LSTMs
    NUM_LAYERS = 1
    BATCH_SIZE = 2
    SRC_LEN = 10  # Length of source sequences
    TRG_LEN = 12  # Length of target sequences

    # ------------------------------
    # 3. Create dummy source and target sequences.
    # ------------------------------
    src_dummy = torch.randint(0, INPUT_DIM, (BATCH_SIZE, SRC_LEN)).to(device)
    trg_dummy = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE, TRG_LEN)).to(device)

    # ------------------------------
    # 4. Initialize the Encoder, Decoder, and Seq2Seq model.
    # ------------------------------
    encoder = Encoder(INPUT_DIM, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
    decoder = Decoder(OUTPUT_DIM, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)

    # ------------------------------
    # 5. Perform a forward pass.
    # ------------------------------
    outputs, attentions = model(src_dummy, trg_dummy)
    print("Output shape:", outputs.shape)  # Expected: (BATCH_SIZE, TRG_LEN, OUTPUT_DIM)
    print("Attention shape:", attentions.shape)  # Expected: (BATCH_SIZE, TRG_LEN, SRC_LEN)
