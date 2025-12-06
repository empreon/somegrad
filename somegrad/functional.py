import numpy as np
from typing import Union, Tuple, Optional, List
from .engine import Tensor, unbroadcast

def exp(input: Tensor) -> Tensor:
    """
    Element-wise exponential function.
    """
    out = Tensor(np.exp(input.data), (input,), 'exp')

    def _backward() -> None:
        # (e^x)' = e^x
        input.grad += unbroadcast(out.data * out.grad, input.data.shape)
    out._backward = _backward
    return out

def log(input: Tensor) -> Tensor:
    """
    Element-wise natural logarithm.
    """
    out = Tensor(np.log(input.data), (input,), 'log')
    
    def _backward() -> None:
        # ln(x)' = 1/x
        input.grad += unbroadcast((1 / input.data) * out.grad, input.data.shape)
    out._backward = _backward
    return out

def abs(input: Tensor) -> Tensor:
    """
    Element-wise absolute value function.
    """
    out = Tensor(np.abs(input.data), (input,), 'abs')

    def _backward() -> None:
        # |x|' = sign(x)
        input.grad += unbroadcast(np.sign(input.data) * out.grad, input.data.shape)
    out._backward = _backward
    return out

def relu(input: Tensor) -> Tensor:
    """
    Rectified Linear Unit activation function.
    f(x) = max(0, x)
    """
    out = Tensor(np.maximum(0, input.data), (input,), 'ReLU')

    def _backward() -> None:
        # Gradient is 1 where data > 0, else 0
        input.grad += unbroadcast((out.data > 0) * out.grad, input.data.shape)
    out._backward = _backward
    return out

def tanh(input: Tensor) -> Tensor:
    """
    Hyperbolic tangent activation function.
    """
    t = (np.exp(2*input.data) - 1)/(np.exp(2*input.data) + 1)
    out = Tensor(t, (input,), 'tanh')

    def _backward() -> None:
        # (tanh(x))' = 1 - tanh(x)^2
        input.grad += unbroadcast((1 - t**2) * out.grad, input.data.shape)
    out._backward = _backward
    return out

def sum(input: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    """
    Sum of array elements over a given axis.
    """
    out = Tensor(input.data.sum(axis=axis, keepdims=keepdims), (input,), 'sum')

    def _backward() -> None:
        grad = out.grad
        
        # If dimensions were collapsed (keepdims=False), restore them for broadcasting
        if not keepdims and axis is not None:
            # Handle single integer axis
            if isinstance(axis, int):
                ax = [axis]
            else:
                ax = list(axis)
            
            # Adjust negative axes
            ndim = input.data.ndim
            ax = [a if a >= 0 else a + ndim for a in ax]
            
            # Expand dims at the correct positions
            for a in sorted(ax):
                grad = np.expand_dims(grad, a)

        input.grad += grad
        
    out._backward = _backward
    return out

def mean(input: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    """
    Compute the arithmetic mean along the specified axis.
    """
    out = Tensor(input.data.mean(axis=axis, keepdims=keepdims), (input,), 'mean')

    def _backward() -> None:
        if axis is None:
            N = input.data.size
        else:
            # Calculate number of elements involved in the mean for the given axis
            N = input.data.size / out.data.size
        
        grad = out.grad / N

        # Same logic as sum: restore dimensions if necessary
        if not keepdims and axis is not None:
            if isinstance(axis, int):
                ax = [axis]
            else:
                ax = list(axis)
            
            ndim = input.data.ndim
            ax = [a if a >= 0 else a + ndim for a in ax]
            
            for a in sorted(ax):
                grad = np.expand_dims(grad, a)
        
        input.grad += grad
        
    out._backward = _backward
    return out

def var(input: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    """
    Compute the variance along the specified axis.
    """
    # var = mean((x - mean(x))**2)
    m = mean(input, axis=axis, keepdims=True)
    return mean((input - m)**2, axis=axis, keepdims=keepdims)

def std(input: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    """
    Compute the standard deviation along the specified axis.
    """
    return var(input, axis=axis, keepdims=keepdims) ** 0.5

def one_hot(input: Tensor, num_classes: int = -1) -> Tensor:
    """
    One-hot encode the tensor.
    """
    # input.data shape: (N, 2) or similar
    N = input.data.shape[0]
    if input.data.ndim == 1:
        out = np.zeros((N, num_classes))
        out[np.arange(N), input.data[:]] = 1
    else:
        input_len = input.data.shape[1] 
    
        # Expected output: (N, input_len * num_classes) -> (N, 54)
        out = np.zeros((N, input_len * num_classes))
        
        for i in range(input_len):
            out[np.arange(N), input.data[:, i] + (i * num_classes)] = 1
        
    return Tensor(out)

def cross_entropy(input: Tensor, target: Union[Tensor, np.ndarray]) -> Tensor:
    """
    Computes the Cross Entropy Loss.
    Assumes 'input' contains logits (unnormalized log-probabilities).
    """
    # Forward Pass
    # For numerical stability subtract maximum value from each row
    max_vals = input.data.max(axis=1, keepdims=True)
    logits = input.data - max_vals 
    
    # Softmax
    counts = np.exp(logits)
    probs = counts / counts.sum(axis=1, keepdims=True)
    
    # Fancy Indexing
    N = input.data.shape[0]
    y_vals = target.data if isinstance(target, Tensor) else target
    
    # Loss Calculation
    # Add small epsilon to prevent log(0)
    correct_confidences = probs[np.arange(N), y_vals] + 1e-8
    loss_val = -np.log(correct_confidences).mean()
    
    # Wrapping
    out = Tensor(loss_val, (input,), 'CrossEntropy')
    
    # Backward Pass
    def _backward() -> None:
        d_logits = probs.copy()
        d_logits[np.arange(N), y_vals] -= 1
        d_logits /= N
        
        input.grad += d_logits * out.grad
        
    out._backward = _backward
    return out

def mse_loss(input: Tensor, target: Union[Tensor, np.ndarray]) -> Tensor:
    """
    Computes the Mean Squared Error Loss.
    """
    target = target if isinstance(target, Tensor) else Tensor(target)
    # We use the composed operations so gradients propagate naturally
    return ((input - target) ** 2).mean()

def reshape(input: Tensor, *shape) -> Tensor:
    """
    Reshape the tensor.
    """
    # Handle both reshape(2, 3) and reshape((2, 3))
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    
    out = Tensor(input.data.reshape(shape), (input,), 'reshape')
    
    def _backward() -> None:
        input.grad += out.grad.reshape(input.data.shape)
    out._backward = _backward
    return out

def histogram(input: Tensor, bins: Union[int, List[float], np.ndarray] = 10, range: Optional[Tuple[float, float]] = None, density: bool = False) -> Tuple[Tensor, Tensor]:
    """
    Computes the histogram of a tensor.
    """
    # detach data for numpy operation
    hist_np, bin_edges_np = np.histogram(input.data, bins=bins, range=range, density=density)
    
    # Wrap results in Tensors.    
    hist = Tensor(hist_np)
    bin_edges = Tensor(bin_edges_np)
    
    return hist, bin_edges