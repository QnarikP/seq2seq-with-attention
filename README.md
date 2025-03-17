# Attention and Self-Attention Project

This project implements a basic attention mechanism and a self-attention module in PyTorch. It is designed for experimentation and visualization, and serves as a foundation for more advanced models (such as transformers).

## Project Structure

```markdown
attention_project/ 
├── data/ 
├── notebooks/
├── src/ 
│ ├── init.py 
│ ├── attention.py 
│ ├── self_attention.py 
│ └── utils.py 
├── tests/ 
│ ├── init.py 
│ └── test_attention.py 
├── README.md 
├── requirements.txt 
└── setup.py
```

## Usage

1. **Data Exploration and Debugging:**  
   Open the Jupyter Notebook in `notebooks/analysis.ipynb` to:
   - Load the translation data (using only the first two columns).
   - Visualize sentence length statistics.
   - Test the self-attention module on dummy input and inspect attention weights.

2. **Running Tests:**  
   Run `pytest` in the root directory to execute tests defined in `tests/test_attention.py`.

3. **Development:**  
   Extend the modules in the **src/** folder as needed. Contributions and improvements are welcome!

## Installation

Install the required packages with:

```bash
pip install -r requirements.txt
```
You can also install the project as a package (if needed):

```bash
python setup.py install
```

# Author
[Qnarik Poghosyan](https://www.linkedin.com/in/qnarik-poghosyan-b4b04a26a/)