import numpy as np

from layers.python_layers.layer import Layer

class SoftMax(Layer):

    def forward(self, input):
        self.p, self.n, self.k = input.shape
        out = np.zeros(input.shape)
        for p in range(self.p):
            for j in range(self.k):
                out[p, :, j] = np.exp(input[p, :, j])/np.sum(np.exp(input[p]), 0)
        return out

    def backward(self, dLdZ):
        return dLdZ