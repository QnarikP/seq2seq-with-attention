"""
Module: utils.py
Description: Utility functions for data loading and visualization.
Author: [Your Name]
Date: [Date]

This module includes functions to load translation data from a text file and to
visualize key properties of the dataset.
"""

import os


def load_translation_data(filepath):
    """
    ================================================================================
    Load Translation Data
    ================================================================================

    Reads a text file containing translation pairs separated by tabs. It extracts
    only the first two columns (assumed to be English and Russian sentences).

    Parameters:
    ------------------------------------------------------------------------------
    filepath : str
               Path to the text file containing translation data.

    Returns:
    ------------------------------------------------------------------------------
    english_sentences : list of str
                        List containing English sentences.
    russian_sentences : list of str
                        List containing Russian sentences.
    ================================================================================
    """
    english_sentences = []
    russian_sentences = []

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                # ------------------------------------------------------------------
                # Clean and split the line.
                # ------------------------------------------------------------------
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                parts = line.split('\t')

                # ------------------------------------------------------------------
                # Ensure the line has at least two columns.
                # ------------------------------------------------------------------
                if len(parts) < 2:
                    continue

                # ------------------------------------------------------------------
                # Extract the first two columns: English and Russian.
                # ------------------------------------------------------------------
                eng = parts[0].strip()
                rus = parts[1].strip()

                english_sentences.append(eng)
                russian_sentences.append(rus)
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")

    return english_sentences, russian_sentences


def visualize_data(english_sentences, russian_sentences):
    """
    ================================================================================
    Visualize Translation Data Properties
    ================================================================================

    This function provides two visualizations:
    1. A bar chart comparing the average sentence length (in characters) of English
       and Russian sentences.
    2. Histograms showing the distribution of sentence lengths for both languages.

    Parameters:
    ------------------------------------------------------------------------------
    english_sentences : list of str
                        List of English sentences.
    russian_sentences : list of str
                        List of Russian sentences.
    ================================================================================
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # ------------------------------------------------------------------
    # Compute sentence lengths.
    # ------------------------------------------------------------------
    eng_lengths = [len(sentence) for sentence in english_sentences]
    rus_lengths = [len(sentence) for sentence in russian_sentences]

    # ------------------------------------------------------------------
    # Calculate average lengths.
    # ------------------------------------------------------------------
    avg_eng_length = np.mean(eng_lengths)
    avg_rus_length = np.mean(rus_lengths)

    # ------------------------------------------------------------------
    # Bar chart: Average sentence lengths.
    # ------------------------------------------------------------------
    labels = ['English', 'Russian']
    averages = [avg_eng_length, avg_rus_length]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, averages, color=['blue', 'red'])
    plt.xlabel('Language')
    plt.ylabel('Average Sentence Length (characters)')
    plt.title('Average Sentence Length Comparison')
    plt.ylim(0, max(averages) + 10)
    plt.show()

    # ------------------------------------------------------------------
    # Histograms: Distribution of sentence lengths.
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.hist(eng_lengths, bins=20, alpha=0.5, label='English', color='blue')
    plt.hist(rus_lengths, bins=20, alpha=0.5, label='Russian', color='red')
    plt.xlabel('Sentence Length (characters)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sentence Lengths')
    plt.legend()
    plt.show()