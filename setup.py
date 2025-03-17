from setuptools import setup, find_packages

setup(
    name='attention_project',
    version='0.1',
    description='A project to experiment with attention and self-attention mechanisms in PyTorch.',
    author='[Qnarik Poghosyan]',
    author_email='[qnarik.poghosyan16@gmail.com]',
    packages=find_packages(),
    install_requires=[
        'torch>=1.8',       # For neural network modules and tensor operations
        'matplotlib',       # For visualization
        'numpy',            # For numerical computations
        'tensorboard'       # For logging with torch.utils.tensorboard
    ],
    entry_points={
        # Optionally, you can define console scripts here for easy command-line access.
        # Example: 'console_scripts': ['run_attention=src.attention:main']
    },
)
