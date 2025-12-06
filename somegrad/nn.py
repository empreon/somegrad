import numpy as np
from somegrad import Tensor

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self):
        return []


class Linear(Module):

    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = Tensor(np.random.randn(fan_in, fan_out)) / fan_in**0.5
        self.bias = Tensor(np.zeros(fan_out)) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d(Module):

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters
        self.gamma = Tensor(np.ones(dim))
        self.beta = Tensor(np.zeros(dim))
        # buffers
        self.running_mean = Tensor(np.zeros(dim))
        self.running_var = Tensor(np.ones(dim)) 

    def __call__(self, x):
        # calculate the forward pass
        if self.training:
            xmean = x.mean(0, keepdims=True) # batch mean
            xvar = x.var(0, keepdims=True) # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
            
        xhat = (x - xmean) / (xvar + self.eps)**0.5 # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        
        # update the buffers
        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:

    def __call__(self, x):
        self.out = x.tanh()
        return self.out

    def parameters(self):
        return []
