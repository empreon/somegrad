import numpy as np
from typing import Union, Tuple, Optional, Set, Callable, List

class Tensor:
    """
    Stores a tensor and its gradient.
    
    This class implements a basic autograd engine using NumPy, supporting
    essential operations for building and training neural networks.
    """

    def __init__(self, data: Union[np.ndarray, float, int, list], _children: Tuple['Tensor', ...] = (), _op: str = ''):
        """
        Initialize the Tensor.

        Args:
            data: The numerical data for the tensor. Can be a scalar, list, or NumPy array.
            _children: A tuple of parent Tensors (used internally for autograd graph construction).
            _op: The operation string that created this Tensor (used for visualization/debugging).
        """
        self.data = np.array(data)
        self.grad = np.zeros_like(self.data, dtype=float)

        self._prev = set(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    def __hash__(self):
        return id(self)

    def __getitem__(self, key) -> 'Tensor':
        """
        Slicing support for the tensor.
        Allows selecting subsets of data, propagating gradients back to the original tensor.
        """
        out = Tensor(self.data[key], (self,), 'slice')

        def _backward() -> None:
            # Add gradient to the specific indices
            np.add.at(self.grad, key, out.grad)
        out._backward = _backward
        return out

    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        Element-wise addition.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward() -> None:
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        Element-wise multiplication.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward() -> None:
            # Product rule: (f*g)' = f'g + fg'
            self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __matmul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        Matrix multiplication (@ operator).
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward() -> None:
            # Gradients for matrix multiplication:
            # C = A @ B  =>  dA = dC @ B.T  and  dB = A.T @ dC
            self.grad += out.grad @ other.data.transpose(-1, -2)
            other.grad += self.data.transpose(-1, -2) @ out.grad
        out._backward = _backward
        return out

    def __pow__(self, other: Union[int, float]) -> 'Tensor':
        """
        Element-wise power function.
        """
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data ** other, (self,), f'**{other}')

        def _backward() -> None:
            # Power rule: n * x^(n-1)
            self.grad += unbroadcast((other * self.data**(other-1)) * out.grad, self.data.shape)
        out._backward = _backward
        return out

    def exp(self) -> 'Tensor':
        """
        Element-wise exponential function.
        """
        from . import functional as F
        return F.exp(self)

    def log(self) -> 'Tensor':
        """
        Element-wise natural logarithm.
        """
        from . import functional as F
        return F.log(self)

    def relu(self) -> 'Tensor':
        """
        Rectified Linear Unit activation function.
        f(x) = max(0, x)
        """
        from . import functional as F
        return F.relu(self)

    def tanh(self) -> 'Tensor':
        """
        Hyperbolic tangent activation function.
        """
        from . import functional as F
        return F.tanh(self)

    def float(self) -> 'Tensor':
        """
        Cast the tensor data to float type.
        """
        return Tensor(self.data.astype(np.float32), (self,), 'float')

    def __float__(self):
        """Float conversion (float(self))"""
        return float(self.data)

    def __int__(self):
        """Int conversion (int(self))"""
        return int(self.data)

    def __lt__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Lesser Than"""
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data < other_data)

    def __gt__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Greater Than"""
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data > other_data)

    def __le__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Lesser Equal Than"""
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data <= other_data)

    def __ge__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Greater Equal Than"""
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data >= other_data)

    def __eq__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Equal With"""
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data == other_data)

    def __ne__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Not Equal With"""
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data != other_data)

    def __neg__(self) -> 'Tensor':  
        """Negation (-self)."""
        return self * -1

    def __abs__(self) -> 'Tensor':
        """Absolute value (abs(self))."""
        from . import functional as F
        return F.abs(self)

    def __radd__(self, other: Union['Tensor', float, int]) -> 'Tensor': 
        """Reverse addition (other + self)."""
        return self + other

    def __sub__(self, other: Union['Tensor', float, int]) -> 'Tensor': 
        """Subtraction (self - other)."""
        return self + (-other)

    def __rsub__(self, other: Union['Tensor', float, int]) -> 'Tensor': 
        """Reverse subtraction (other - self)."""
        return other + (-self)

    def __rmul__(self, other: Union['Tensor', float, int]) -> 'Tensor': 
        """Reverse multiplication (other * self)."""
        return self * other

    def __truediv__(self, other: Union['Tensor', float, int]) -> 'Tensor': 
        """Division (self / other)."""
        return self * other**-1

    def __rtruediv__(self, other: Union['Tensor', float, int]) -> 'Tensor': 
        """Reverse division (other / self)."""
        return other * self**-1

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"

    # Helper Functions
    
    def backward(self) -> None:
        """
        Performs backpropagation starting from this tensor.
        Constructs the topological graph of the computational history and applies the chain rule.
        Sets the gradient of the starting tensor to 1.0 (assuming it's a scalar loss).
        """
        topo: List['Tensor'] = []
        visited: Set['Tensor'] = set()
        
        def build_topo(v: 'Tensor') -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        # Last gradient needs to be 1.
        self.grad = np.ones_like(self.data)
        
        for node in reversed(topo):
            node._backward()

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """
        Sum of array elements over a given axis.
        """
        from . import functional as F
        return F.sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """
        Compute the arithmetic mean along the specified axis.
        """
        from . import functional as F
        return F.mean(self, axis=axis, keepdims=keepdims)

    def var(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """
        Compute the variance along the specified axis.
        """
        from . import functional as F
        return F.var(self, axis=axis, keepdims=keepdims)

    def std(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """
        Compute the standard deviation along the specified axis.
        """
        from . import functional as F
        return F.std(self, axis=axis, keepdims=keepdims)

    def one_hot(self, num_classes: int = -1) -> 'Tensor':
        """
        One-hot encode the tensor.
        """
        from . import functional as F
        return F.one_hot(self, num_classes=num_classes)

    def cross_entropy(self, y: Union['Tensor', np.ndarray]) -> 'Tensor':
        """
        Computes the Cross Entropy Loss.
        """
        from . import functional as F
        return F.cross_entropy(self, y)

    def reshape(self, *shape) -> 'Tensor':
        """
        Reshape the tensor.
        Usage: x.reshape(2, 3) or x.reshape((2, 3))
        """
        from . import functional as F
        return F.reshape(self, *shape)

def unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Handles NumPy broadcasting issues during backpropagation.
    If the gradient shape doesn't match the original data shape, sum out the broadcasted dimensions.
    """
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