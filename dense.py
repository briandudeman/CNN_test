import math
import numpy as np

from layer import Layer


np.random.seed(6)


class FCLayer(Layer):
    def __init__(self, num_nodes):
        # being fed through(batch size)
        self.m = num_nodes  # number of nodes

    def forward(self, input):
      self.input = input
      self.p, self.d, self.n = input.shape  # p is batch size, d is number of dimensions to the data and n is the number of data points

      self.x_limit = math.sqrt((2 / (self.n + self.m)))
      self.weights = np.random.uniform(-self.x_limit, self.x_limit, size=(self.m, self.d))  # d by m
      self.biases = np.random.uniform(-self.x_limit, self.x_limit, size=(self.m, 1))

      self.z = np.zeros((self.p, self.m, self.n))
      for p in range(self.p):
        self.z[p] = (self.weights @ self.input[p]) + self.biases
      return self.z
    
    def backward(self, dLdZ, lr=.00005):
      self.dLdA = self.weights.T @ dLdZ

      self.dLdW = np.array([dLdZ[i] @ self.input[i].T for i in range(self.input.shape[0])])
      self.dLdW = np.sum(self.dLdW, axis=0)
      self.dLdW0 = np.sum(dLdZ, axis=0)  # m by n (same size as dLdZ)
      #print("dldw", self.dLdW)
      #print("shape of dLdW0, in linear", np.shape(self.dLdW0))
      #print("shape of dLdW, in linear", np.shape(self.dLdW))
      self.weights -= lr*self.dLdW
      self.biases -= lr*self.dLdW0
      return self.dLdA  # d by n
