# K-Means and SymNMF

Implementation of K-Means and Symmetric Nonnegative Matrix Factorization (SymNMF) clustering algorithms using Python and C extensions.

This project focuses on unsupervised machine learning, matrix factorization, clustering, and efficient numerical computations.

---

## Technologies
- Python
- C
- NumPy
- Machine Learning
- Matrix Optimization

---

## Project Structure

### kmeans.py
Implementation of the K-Means clustering algorithm:
- Euclidean distance calculation
- Cluster assignment
- Centroid recomputation
- Iterative convergence process

### symnmf.py
Python interface for Symmetric Nonnegative Matrix Factorization:
- Loads datasets
- Initializes matrices
- Executes SymNMF optimization
- Handles command-line execution

### symnmf.c
Core C implementation of:
- Similarity matrix computation
- Diagonal degree matrix
- Normalized similarity matrix
- SymNMF optimization updates
- Matrix operations

### symnmfmodule.c
Python C-extension module:
- Connects Python with the C implementation
- Converts NumPy arrays into C matrices
- Exposes C functions to Python

### symnmf.h
Header file containing declarations for matrix and SymNMF functions.

### setup.py
Build configuration for the Python C-extension module.

### Makefile
Compilation and build automation.

---

## Features
- K-Means clustering
- SymNMF implementation
- Python-C integration
- Matrix optimization algorithms
- Efficient numerical computations
- Unsupervised learning techniques

---

## How to Run

```bash
python setup.py build_ext --inplace
python symnmf.py <k> <goal> <input_file>
