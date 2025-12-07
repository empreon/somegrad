import numpy as np

class SGD:
    """Stochastic Gradient Descent Optimizer"""
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        """Update parameters according to gradient"""
        for p in self.params:
            if p.grad is None:
                continue
            p.buffer.data -= self.lr * p.grad

    def zero_grad(self):
        """Zeor the gradients"""
        for p in self.params:
            p.grad = None
