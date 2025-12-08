# Somegrad

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Status](https://img.shields.io/badge/status-active_development-green)

Somegrad is a lightweight deep learning framework implemented from scratch to understand the internal mechanics of modern AI systems. It is inspired by Andrej Karpathy's **micrograd** and George Hotz's **tinygrad**.

Unlike high-level wrappers, Somegrad focuses on the engine level: managing computational graphs, implementing backpropagation manually, and handling memory buffers across different devices.

## Architecture

The framework is designed with a "Systems-First" approach:

```mermaid
graph TD;
    User_Code --> Tensor_API;
    Tensor_API --> Computational_Graph;
    Tensor_API -- Dispatch --> Device_Backend;
    Device_Backend --> CPU_Buffer["CPU Buffer (NumPy)"];
    Device_Backend -.-> GPU_Buffer["GPU Buffer (CUDA) - WIP"];
````

  * **`tensor.py`**: User-facing API that builds the DAG (Directed Acyclic Graph).
  * **`functional.py`**: Stateless implementations of mathematical operations and their backward passes.
  * **`device/`**: Low-level buffer management. Isolated from the autograd logic to allow easy swapping of compute backends (e.g., CPU to CUDA).

## Features

  * **Autograd Engine:** Implements a topological sort-based automatic differentiation system (Reverse-mode AD).
  * **Device Abstraction:** Modular backend architecture separating API (`Tensor`) from execution (`Buffer`), currently supporting CPU (NumPy) with plans for CUDA.
  * **Robust Broadcasting:** Correctly handles gradient accumulation for broadcasted operations using unbroadcast logic.
  * **Neural Network Modules:** Includes `Linear`, `BatchNorm1d` (with running stats for train/eval modes), and various activation functions (`ReLU`, `Tanh`, `Softmax`).
  * **Optimizers:** Modular `SGD` optimizer implementation.

## Installation

Clone the repository and install the dependencies:

```bash
git clone [https://github.com/username/somegrad.git](https://github.com/username/somegrad.git)
cd somegrad
pip install -r requirements.txt
```

To install the package in editable mode:

```bash
pip install -e .
```

## Examples

The `examples/` directory contains Jupyter notebooks demonstrating the capabilities of the framework:

  * **01\_linear\_regression.ipynb:** Basic implementation of linear regression using manual gradient updates.
  * **02\_trigram\_language\_model.ipynb:** A character-level language model using a simple lookup table and cross-entropy loss.
  * **03\_mlp\_language\_model.ipynb:** A Multi-Layer Perceptron (MLP) based language model, introducing hidden layers and non-linearities.
  * **04\_mlp2\_language\_model.ipynb:** A deeper MLP implementation with Batch Normalization and more complex optimization dynamics.

## Roadmap

The project is evolving from a Python-only implementation to a high-performance system:

  - [x] Core Autograd Engine (Topological Sort, DAG)
  - [x] NumPy Backend (CPU)
  - [x] NN Modules (Linear, BatchNorm, Activations)
  - [ ] CUDA Backend: Implementing custom CUDA kernels for matrix operations.
  - [ ] Lazy Evaluation: Moving from eager execution to graph compilation for kernel fusion optimization.
