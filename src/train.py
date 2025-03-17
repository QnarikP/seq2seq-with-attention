"""
Module: train.py
Description: Training script for the Seq2Seq model with attention/self-attention for English→Russian translation.
Author: [Your Name]
Date: [Date]

This script:
- Loads and preprocesses the translation data.
- Builds simple vocabularies for source (English) and target (Russian).
- Defines a PyTorch Dataset and DataLoader.
- Instantiates the Seq2Seq model (using Encoder and Decoder).
- Sets up the loss function, optimizer, and tensorboard logging.
- Runs a training loop with teacher forcing and saves model checkpoints.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Import our Seq2Seq model and related modules.
from src.seq2seq import Encoder, Decoder, Seq2Seq
from src.utils import load_translation_data


# ================================================================================
#                     Vocabulary Building and Data Preprocessing
# ================================================================================
def build_vocab(sentences, min_freq=1):
    """
    Build a simple vocabulary mapping from tokens to indices.
    Adds special tokens: <pad>, <sos>, <eos>, <unk>.
    """
    from collections import Counter
    special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
    token_counts = Counter()

    for sentence in sentences:
        tokens = sentence.lower().split()
        token_counts.update(tokens)

    # Include tokens that appear at least min_freq times.
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    idx = len(special_tokens)
    for token, count in token_counts.items():
        if count >= min_freq:
            vocab[token] = idx
            idx += 1
    return vocab


def numericalize(sentence, vocab):
    """
    Convert a sentence (string) into a list of token indices.
    Adds <sos> at the beginning and <eos> at the end.
    """
    tokens = sentence.lower().split()
    indices = [vocab.get('<sos>')]
    indices += [vocab.get(token, vocab.get('<unk>')) for token in tokens]
    indices.append(vocab.get('<eos>'))
    return indices


# ================================================================================
#                     Custom Dataset for Translation Data
# ================================================================================
class TranslationDataset(Dataset):
    """
    Custom dataset for English→Russian translation.
    Expects lists of English and Russian sentences.
    """

    def __init__(self, eng_sentences, rus_sentences, src_vocab, trg_vocab, max_len=50):
        self.eng_sentences = eng_sentences
        self.rus_sentences = rus_sentences
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_len = max_len  # Optional max length for padding/truncation

    def __len__(self):
        return len(self.eng_sentences)

    def __getitem__(self, idx):
        src = numericalize(self.eng_sentences[idx], self.src_vocab)
        trg = numericalize(self.rus_sentences[idx], self.trg_vocab)
        # Truncate if necessary
        src = src[:self.max_len]
        trg = trg[:self.max_len]
        return torch.tensor(src), torch.tensor(trg)


def collate_fn(batch):
    """
    Collate function to pad sequences in the batch.
    """
    from torch.nn.utils.rnn import pad_sequence
    src_batch, trg_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)  # <pad> index assumed 0
    trg_batch = pad_sequence(trg_batch, batch_first=True, padding_value=0)
    return src_batch, trg_batch


# ================================================================================
#                          Training Configuration
# ================================================================================
def train_model():
    # ------------------------------
    # Hyperparameters and settings.
    # ------------------------------
    DATA_PATH = os.path.join('..', 'data', 'rus.txt')  # Adjust path as needed.
    NUM_EPOCHS = 1000
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    TEACHER_FORCING_RATIO = 0.5
    MAX_LEN = 50  # Maximum sentence length (for padding/truncation)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------------
    # Load raw translation data.
    # ------------------------------
    # Load raw translation data.
    print("Loading translation data...")
    eng_sentences, rus_sentences = load_translation_data(DATA_PATH)
    print(f"Loaded {len(eng_sentences)} sentence pairs.")

    # Select only half of the dataset
    half_size = len(eng_sentences) // 5
    eng_sentences = eng_sentences[:half_size]
    rus_sentences = rus_sentences[:half_size]
    print(f"Using {len(eng_sentences)} sentence pairs for training.")

    # ------------------------------
    # Build vocabularies.
    # ------------------------------
    print("Building vocabularies...")
    src_vocab = build_vocab(eng_sentences)
    trg_vocab = build_vocab(rus_sentences)
    src_vocab_size = len(src_vocab)
    trg_vocab_size = len(trg_vocab)
    print(f"Source vocabulary size: {src_vocab_size}")
    print(f"Target vocabulary size: {trg_vocab_size}")


    # ------------------------------
    # Create Dataset and DataLoader.
    # ------------------------------
    dataset = TranslationDataset(eng_sentences, rus_sentences, src_vocab, trg_vocab, max_len=MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    print("Data loaded.")
    # ------------------------------
    # Model Hyperparameters.
    # ------------------------------
    EMBED_DIM = 256
    HIDDEN_DIM = 512
    NUM_LAYERS = 1

    # ------------------------------
    # Initialize Encoder, Decoder, and Seq2Seq Model.
    # ------------------------------
    encoder = Encoder(input_dim=src_vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    decoder = Decoder(output_dim=trg_vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    model = Seq2Seq(encoder, decoder, device).to(device)
    print("Model created.")
    # ------------------------------
    # Define Loss Function and Optimizer.
    # ------------------------------
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming <pad> token index is 0.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ------------------------------
    # TensorBoard Setup for Logging.
    # ------------------------------
    writer = SummaryWriter(log_dir='runs/seq2seq_training')

    # ------------------------------
    # Training Loop.
    # ------------------------------
    print("Training...")
    model.train()
    global_step = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss = 0
        start_time = time.time()
        print("Epoch {}/{}".format(epoch, NUM_EPOCHS))
        for batch_idx, (src_batch, trg_batch) in enumerate(dataloader):
            if batch_idx % 100 == 0:
                print("Batch {}/{}".format(batch_idx, len(dataloader)))

            src_batch = src_batch.to(device)
            trg_batch = trg_batch.to(device)

            optimizer.zero_grad()

            # Forward pass through the model.
            output, _ = model(src_batch, trg_batch, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
            # output shape: [batch_size, trg_len, trg_vocab_size]
            # Reshape outputs and targets for loss computation.
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)  # Exclude <sos> token output.
            trg = trg_batch[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            writer.add_scalar('Loss/Batch', loss.item(), global_step)
            global_step += 1

        avg_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch}/{NUM_EPOCHS} - Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s")
        writer.add_scalar('Loss/Epoch', avg_loss, epoch)

        # ------------------------------
        # Save the model checkpoint.
        # ------------------------------
        checkpoint_path = os.path.join('..', 'models', f'seq2seq_checkpoint_{epoch}.pth')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

    writer.close()


if __name__ == "__main__":
    print("Using device: ", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    train_model()