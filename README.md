# English-to-Russian Machine Translation with Seq2Seq and Attention

This repository implements an English-to-Russian translation system using a Sequence-to-Sequence (Seq2Seq) model with attention mechanisms. The project is built with PyTorch and includes modules for data preprocessing, model definition, training, evaluation, and attention.

## Project Structure

- **notebooks/**
  - **analysis.ipynb**
  
    Testing attention mechanism.
- **src/**
  - **seq2seq.py**  
    Implements the core Seq2Seq model with an LSTM-based Encoder and an Attention-based Decoder.
  - **attention.py**  
    Provides a basic scaled dot-product attention mechanism.
  - **self_attention.py**  
    Implements self-attention for projecting inputs into query, key, and value spaces.
  - **train.py**  
    Contains the training pipeline: data loading, vocabulary building, model training, and checkpoint saving.
  - **evaluation.py**  
    Evaluates the trained model on test data and demonstrates sample translations.
  - **utils.py**  
    Includes utility functions for data loading, preprocessing, and visualization.
- **data/**  
  Contains the translation dataset (e.g., `rus.txt`).

## Features

- Preprocessing of English–Russian sentence pairs.
- Dynamic vocabulary building for source and target languages.
- LSTM-based encoder and attention-enabled decoder.
- End-to-end training and evaluation pipelines.
- Model checkpointing.

## Requirements

   ```bash
   pip install requirements
   ```

## Usage

1. **Data Preparation**:  
   Place your tab-separated translation dataset in the `data/` directory.

2. **Training**:  
   Run the training script to build vocabularies, train the model, and save checkpoints.
   ```bash
   python src/train.py
   ```

3. **Evaluation**:  
   Evaluate the model and test translation inference.
   ```bash
   python src/evaluation.py
   ```