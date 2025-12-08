import numpy as np
from typing import Tuple

def unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    if grad.shape != shape:
        # Normal broadcasting: Handle extra leading dimensions
        # e.g., gradient has shape (3, 4), original was (4,) -> sum over axis 0
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        
        # Handle dimensions that were broadcasted to match
        # e.g., gradient (3, 4), original (3, 1) -> sum over axis 1
        for i, dim in enumerate(shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        
        # If dimensions match but not shapes (e.g., (1,1) -> scalar), reshape
        if np.prod(grad.shape) == np.prod(shape):
            grad = grad.reshape(shape)
                
    return grad