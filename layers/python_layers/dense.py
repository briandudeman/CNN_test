import math
import numpy as np

from layers.python_layers.weight_layer import WeightLayer


np.random.seed(6)


class FCLayer(WeightLayer):
    def __init__(self, num_nodes):
      # being fed through(batch size)
      self.m = num_nodes  # number of nodes
      self.weights = None
      self.biases = None
      super().__init__()

    def forward(self, input):
      self.input = input
      #print("input shape", input.shape)
      self.p, self.d, self.n = input.shape  # p is batch size, d is number of dimensions to the data and n is the number of data points

      if not self.weights:
        self.x_limit = math.sqrt((2 / (self.n + self.m)))
        self.weights = np.random.uniform(-self.x_limit, self.x_limit, size=(self.m, self.d))  # d by m
        self.biases = np.random.uniform(-self.x_limit, self.x_limit, size=(self.m, 1))

      #print("weights", self.weights)
      #print("biases", self.biases)
      self.z = np.zeros((self.p, self.m, self.n))
      for p in range(self.p):
        self.z[p] = (self.weights @ self.input[p]) + self.biases
      return self.z
    
    def backward(self, dLdZ, step, lr=.00001):
      print("\n dLdZ shape", dLdZ.shape)
      print("weights shape", self.weights.shape)
      print("input shape", self.input.shape)
      print("biases shape", self.biases.shape)
      
      self.dLdA = self.weights.T @ dLdZ

      self.dLdW = np.array([dLdZ[i] @ self.input[i].T for i in range(self.input.shape[0])])
      self.dLdW = np.sum(self.dLdW, axis=0)
      # gradients for biases should sum over batch and feature map dims, result shape (m,1)
      self.dLdW0 = np.sum(dLdZ, axis=0)
      #print("dldw", self.dLdW)
      #print("shape of dLdW0, in linear", np.shape(self.dLdW0))
      #print("shape of dLdW, in linear", np.shape(self.dLdW))
      self.weights -= self.optimizer_weights.backward(self.dLdW, step)
      self.biases -= self.optimizer_biases.backward(self.dLdW0, step)

      print("dLdA shape", self.dLdA.shape)
      print("dLdW shape", self.dLdW.shape)
      print("dLdW0 shape", self.dLdW0.shape)
      return self.dLdA  # d by n
