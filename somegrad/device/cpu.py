import numpy as np
from typing import Tuple
from .base import Buffer

class CPUBuffer(Buffer):

    def __init__(self, data):

        if not isinstance(data, np.ndarray):
            data = np.array(data)
        super().__init__(data)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def __add__(self, other) -> 'CPUBuffer':
        other_data = other.data if isinstance(other, CPUBuffer) else other
        return CPUBuffer(self.data + other_data)

    def __mul__(self, other) -> 'CPUBuffer':
        other_data = other.data if isinstance(other, CPUBuffer) else other
        return CPUBuffer(self.data * other_data)

    def __matmul__(self, other) -> 'CPUBuffer':
        other_data = other.data if isinstance(other, CPUBuffer) else other
        return CPUBuffer(self.data @ other_data)
    
    def __pow__(self, other) -> 'CPUBuffer':
        other_data = other.data if isinstance(other, CPUBuffer) else other
        return CPUBuffer(self.data ** other_data)

    def exp(self) -> 'CPUBuffer':
        return CPUBuffer(np.exp(self.data))

    def log(self) -> 'CPUBuffer':
        return CPUBuffer(np.log(self.data))

    def log10(self) -> 'CPUBuffer':
        return CPUBuffer(np.log10(self.data))

    def relu(self) -> 'CPUBuffer':
        return CPUBuffer(np.maximum(0, self.data))

    def tanh(self) -> 'CPUBuffer':
        return CPUBuffer(np.tanh(self.data))

    def __neg__(self) -> 'CPUBuffer':
        return CPUBuffer(-self.data)

    def __abs__(self) -> 'CPUBuffer':
        return CPUBuffer(np.abs(self.data))

    def __radd__(self, other) -> 'CPUBuffer':
        return self + other

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other) -> 'CPUBuffer':
        return other + (-self)
    
    def __rmul__(self, other) -> 'CPUBuffer':
        return self * other

    def __truediv__(self, other) -> 'CPUBuffer':
        return self * other**-1
    
    def __rtruediv__(self, other) -> 'CPUBuffer':
        return other * self**-1
    
    def __rpow__(self, other) -> 'CPUBuffer':
        return other ** self.data
    
    def __eq__(self, other) -> 'CPUBuffer':
        return self.data == other

    def __ne__(self, other) -> 'CPUBuffer':
        return self.data != other
    
    def __lt__(self, other) -> 'CPUBuffer':
        return self.data < other
    
    def __gt__(self, other) -> 'CPUBuffer':
        return self.data > other
    
    def __le__(self, other) -> 'CPUBuffer':
        return self.data <= other
    
    def __ge__(self, other) -> 'CPUBuffer':
        return self.data >= other