import numpy as np
from typing import Tuple, List, Set
from .device import CPUBuffer
from .utils import unbroadcast

class Tensor:

    def __init__(self, data, device='cpu', _children=(), _op=''):
        self.device = device

        if isinstance(data, CPUBuffer):
            self.buffer = data
        elif device == 'cpu':
            self.buffer = CPUBuffer(data)
        else:
            raise ValueError(f"Unsupported device: {device}")

        self.grad = None
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __hash__(self):
        return id(self)

    @property
    def data(self) -> np.ndarray:
        return self.buffer.data

    @data.setter
    def data(self, new_data):
        self.buffer.data = new_data

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.buffer.shape

    def __add__(self, other) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        out_buffer = self.buffer + other.buffer
        out = Tensor(out_buffer, device=self.device, _children=(self, other), _op='+')

        def _backward() -> None:
            if self.grad is None: self.grad = np.zeros(self.shape, dtype=np.float32)
            if other.grad is None: other.grad = np.zeros(other.shape, dtype=np.float32)

            self.grad += unbroadcast(out.grad, self.shape)
            other.grad += unbroadcast(out.grad, other.shape)
        out._backward = _backward
        return out
    
    def __mul__(self, other) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        out_buffer = self.buffer * other.buffer
        out = Tensor(out_buffer, device=self.device, _children=(self, other), _op='*')
        
        def _backward() -> None:
            if self.grad is None: self.grad = np.zeros(self.shape, dtype=np.float32)
            if other.grad is None: other.grad = np.zeros(other.shape, dtype=np.float32)

            self.grad += unbroadcast(out.grad * other.data, self.shape)
            other.grad += unbroadcast(out.grad * self.data, other.shape)
        out._backward = _backward
        return out
    
    def __matmul__(self, other) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        out_buffer = self.buffer @ other.buffer
        out = Tensor(out_buffer, device=self.device, _children=(self, other), _op='@')

        def _backward() -> None:
            if self.grad is None: self.grad = np.zeros(self.shape, dtype=np.float32)
            if other.grad is None: other.grad = np.zeros(other.shape, dtype=np.float32)

            self.grad += unbroadcast(out.grad @ other.data.transpose(-1, -2), self.shape)
            other.grad += unbroadcast(self.data.transpose(-1, -2) @ out.grad, other.shape)
        out._backward = _backward
        return out
    
    def __pow__(self, other) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        out_buffer = self.buffer ** other.buffer
        out = Tensor(out_buffer, device=self.device, _children=(self, other), _op='**')
        
        def _backward() -> None:
            if self.grad is None: self.grad = np.zeros(self.shape, dtype=np.float32)
            if other.grad is None: other.grad = np.zeros(other.shape, dtype=np.float32)

            self.grad += unbroadcast(out.grad * other.data * self.data ** (other.data - 1), self.shape)
            other.grad += unbroadcast(out.grad * self.data * np.log(self.data), other.shape)
        out._backward = _backward
        return out

    def __neg__(self) -> 'Tensor':
        return self * -1

    def __radd__(self, other) -> 'Tensor':
        return self + other

    def __sub__(self, other) -> 'Tensor': 
        return self + (-other)

    def __rsub__(self, other) -> 'Tensor': 
        return other + (-self)

    def __rmul__(self, other) -> 'Tensor': 
        return self * other

    def __truediv__(self, other) -> 'Tensor': 
        return self * other**-1

    def __rtruediv__(self, other) -> 'Tensor': 
        return other * self**-1

    def __eq__(self, other) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        return Tensor(self.buffer == other.buffer, device=self.device)

    def __ne__(self, other) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        return Tensor(self.buffer != other.buffer, device=self.device)

    def __lt__(self, other) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        return Tensor(self.buffer < other.buffer, device=self.device)

    def __gt__(self, other) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        return Tensor(self.buffer > other.buffer, device=self.device)

    def __le__(self, other) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        return Tensor(self.buffer <= other.buffer, device=self.device)

    def __ge__(self, other) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        return Tensor(self.buffer >= other.buffer, device=self.device)

    def __abs__(self) -> 'Tensor':
        from . import functional as F
        return F.abs(self)

    def exp(self) -> 'Tensor':
        from . import functional as F
        return F.exp(self)

    def log(self) -> 'Tensor':
        from . import functional as F
        return F.log(self)

    def log10(self) -> 'Tensor':
        from . import functional as F
        return F.log10(self)

    def relu(self) -> 'Tensor':
        from . import functional as F
        return F.relu(self)

    def tanh(self) -> 'Tensor':
        from . import functional as F
        return F.tanh(self)

    def sum(self, axis=None, keepdims=False) -> 'Tensor':
        from . import functional as F
        return F.sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False) -> 'Tensor':
        from . import functional as F
        return F.mean(self, axis=axis, keepdims=keepdims)

    def var(self, axis=None, keepdims=False) -> 'Tensor':
        from . import functional as F
        return F.var(self, axis=axis, keepdims=keepdims)

    def std(self, axis=None, keepdims=False) -> 'Tensor':
        from . import functional as F
        return F.std(self, axis=axis, keepdims=keepdims)

    def reshape(self, *shape) -> 'Tensor':
        from . import functional as F
        return F.reshape(self, *shape)

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"

    # Helper Functions

    def backward(self) -> None:
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