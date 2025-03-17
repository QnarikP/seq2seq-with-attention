"""
Module: evaluation.py
Description: Evaluation script for the Seq2Seq model with attention/self-attention.
Author: [Your Name]
Date: [Date]

This script:
- Loads the trained Seq2Seq model checkpoint.
- Loads and preprocesses the translation test data.
- Evaluates the model on the test set and computes metrics (e.g., loss, and a placeholder BLEU score).
- Demonstrates translation inference on sample sentences.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.seq2seq import Encoder, Decoder, Seq2Seq
from src.utils import load_translation_data
from train import build_vocab, numericalize


# ================================================================================
#                     Custom Dataset for Evaluation
# ================================================================================
class TranslationDatasetEval(torch.utils.data.Dataset):
    """
    Custom dataset for evaluation.
    """

    def __init__(self, eng_sentences, rus_sentences, src_vocab, trg_vocab, max_len=50):
        self.eng_sentences = eng_sentences
        self.rus_sentences = rus_sentences
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.eng_sentences)

    def __getitem__(self, idx):
        src = numericalize(self.eng_sentences[idx], self.src_vocab)
        trg = numericalize(self.rus_sentences[idx], self.trg_vocab)
        src = src[:self.max_len]
        trg = trg[:self.max_len]
        return torch.tensor(src), torch.tensor(trg)


def collate_fn(batch):
    """
    Collate function to pad sequences.
    """
    from torch.nn.utils.rnn import pad_sequence
    src_batch, trg_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    trg_batch = pad_sequence(trg_batch, batch_first=True, padding_value=0)
    return src_batch, trg_batch


# ================================================================================
#                         Evaluation Functionality
# ================================================================================
def evaluate_model():
    # ------------------------------
    # Configuration and Hyperparameters.
    # ------------------------------
    DATA_PATH = os.path.join('..', 'data', 'rus.txt')  # Test data file.
    MAX_LEN = 50
    BATCH_SIZE = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------------
    # Load the evaluation data.
    # ------------------------------
    print("Loading evaluation data...")
    eng_sentences, rus_sentences = load_translation_data(DATA_PATH)
    print(f"Loaded {len(eng_sentences)} sentence pairs for evaluation.")
    half_size = len(eng_sentences) // 5
    eng_sentences = eng_sentences[:half_size]
    rus_sentences = rus_sentences[:half_size]

    # ------------------------------
    # Build vocabularies (should match those used during training).
    # In practice, you would load these from disk.
    # ------------------------------
    src_vocab = build_vocab(eng_sentences)
    trg_vocab = build_vocab(rus_sentences)
    src_vocab_size = len(src_vocab)
    trg_vocab_size = len(trg_vocab)

    # ------------------------------
    # Create Dataset and DataLoader for evaluation.
    # ------------------------------
    dataset = TranslationDatasetEval(eng_sentences, rus_sentences, src_vocab, trg_vocab, max_len=MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    print("Data Loaded.")
    # ------------------------------
    # Model Hyperparameters (should match training).
    # ------------------------------
    EMBED_DIM = 256
    HIDDEN_DIM = 512
    NUM_LAYERS = 1

    # ------------------------------
    # Initialize the model.
    # ------------------------------
    encoder = Encoder(input_dim=src_vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    decoder = Decoder(output_dim=trg_vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    model = Seq2Seq(encoder, decoder, device).to(device)
    print("Model loaded.")
    # ------------------------------
    # Load the trained model checkpoint.
    # ------------------------------
    checkpoint_path = os.path.join('..', 'models', 'seq2seq_checkpoint_100.pth')
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model checkpoint from {checkpoint_path}")
    else:
        print("Checkpoint not found. Exiting evaluation.")
        return

    # Set model to evaluation mode.
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    total_loss = 0
    with torch.no_grad():
        for src_batch, trg_batch in dataloader:
            src_batch = src_batch.to(device)
            trg_batch = trg_batch.to(device)

            output, _ = model(src_batch, trg_batch, teacher_forcing_ratio=0.0)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg_batch[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Average Evaluation Loss: {avg_loss:.4f}")

    # ------------------------------
    # Inference: Translate a few sample sentences.
    # ------------------------------
    sample_idxs = [5, 10, 20, 60, 70, 100]
    for sample_idx in sample_idxs:
        sample_src = eng_sentences[sample_idx]
        print("\nSample Translation Inference:")
        print("Source (English):", sample_src)

        # Convert sample to tensor.
        sample_tensor = torch.tensor(numericalize(sample_src, src_vocab)).unsqueeze(0).to(device)
        model.eval()

        # Create an empty target sequence with only <sos>.
        trg_indexes = [trg_vocab.get('<sos>')]
        max_trg_len = MAX_LEN
        for _ in range(max_trg_len):
            trg_tensor = torch.tensor(trg_indexes).unsqueeze(0).to(device)
            with torch.no_grad():
                output, _ = model(sample_tensor, trg_tensor, teacher_forcing_ratio=0.0)
            next_token = output[0, -1, :].argmax().item()
            trg_indexes.append(next_token)
            if next_token == trg_vocab.get('<eos>'):
                break

        # Convert indices back to words.
        inv_trg_vocab = {idx: token for token, idx in trg_vocab.items()}
        translated_sentence = " ".join([inv_trg_vocab.get(idx, '<unk>') for idx in trg_indexes])
        print("Translated (Russian):", translated_sentence)


if __name__ == "__main__":
    evaluate_model()