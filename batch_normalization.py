import numpy as np
import math

from layer import Layer

class BatchNormalization(Layer):

    def forward(self, input, epsilon=.0001):
        self.input = input
        self.epsilon = epsilon
        
        self.x_limit = math.sqrt((2 / sum(input.shape[1:])))      
        self.gamma = np.random.uniform(-self.x_limit, self.x_limit, size=self.input.shape)
        self.beta = np.random.uniform(-self.x_limit, self.x_limit, size=self.input.shape)


        self.mean = np.average(input, axis=0)
        self.variance = np.var(input, axis=0)
        
        self.normalized = (self.input - self.mean) / np.sqrt((self.variance + epsilon))
        
        self.out = np.multiply(self.normalized, self.gamma) + self.beta
        return self.out


    def backward(self, dLdZ):
        self.dLdX = np.zeros(self.input.shape)
        self.dLdG = np.multiply(self.normalized, dLdZ)
        self.dLdB = dLdZ.copy()

        self.dLdVar = np.multiply(dLdZ, self.input - self.mean) * ((-self.gamma / 2) * (self.variance + self.epsilon)**(-3/2))
        self.dLdMu = np.multiply(dLdZ, (-self.gamma / np.sqrt(self.variance + self.epsilon))) + self.dLdVar * (-2 / self.input.shape[0]) * np.sum(self.input - self.mean, axis=0)
        self.dLdXhat = np.multiply(dLdZ, self.gamma)

        self.dLdX = self.dLdXhat * (1 / np.sqrt(self.mean + self.epsilon)) + self.dLdVar * ((2 * (self.input - self.mean)) / self.input.shape[0]) + self.dLdMu * (1 / self.input.shape[0])

        return self.dLdX


