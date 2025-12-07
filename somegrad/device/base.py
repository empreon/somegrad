from typing import Tuple
import numpy as np

class Buffer:

    def __init__(self, data):
        self.data = data

    def __add__(self, other): raise NotImplementedError
    def __mul__(self, other): raise NotImplementedError
    def __matmul__(self, other): raise NotImplementedError
    def __pow__(self, other): raise NotImplementedError

    def exp(self): raise NotImplementedError
    def log(self): raise NotImplementedError
    def log10(self): raise NotImplementedError
    def relu(self): raise NotImplementedError
    def tanh(self): raise NotImplementedError

    def __neg__(self): raise NotImplementedError
    def __abs__(self): raise NotImplementedError
    def __radd__(self, other): raise NotImplementedError
    def __sub__(self, other): raise NotImplementedError
    def __rsub__(self, other): raise NotImplementedError
    def __rmul__(self, other): raise NotImplementedError
    def __truediv__(self, other): raise NotImplementedError
    def __rtruediv__(self, other): raise NotImplementedError
    def __rpow__(self, other): raise NotImplementedError

    def __eq__(self, other): raise NotImplementedError
    def __ne__(self, other): raise NotImplementedError
    def __lt__(self, other): raise NotImplementedError
    def __gt__(self, other): raise NotImplementedError
    def __le__(self, other): raise NotImplementedError
    def __ge__(self, other): raise NotImplementedError

    @property
    def shape(self) -> Tuple[int, ...]: raise NotImplementedError

    def numpy(self) -> np.ndarray: raise NotImplementedError