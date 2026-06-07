import numpy as np
import math

from layers.python_layers.layer import Layer

class BatchNormalization(Layer):

    def __init__(self):
        self.first_time_called = False

    def forward(self, input, epsilon=.0001):
        #print("batch norm input shape", input.shape)
        self.input = input
        self.epsilon = epsilon
        
        if not self.first_time_called:
            self.first_time_called = True
            self.x_limit = math.sqrt((2 / sum(input.shape[1:])))      
            self.gamma = np.random.uniform(-self.x_limit, self.x_limit, size=self.input.shape[1:])
            self.beta = np.random.uniform(-self.x_limit, self.x_limit, size=self.input.shape[1:])


        self.mean = np.average(input, axis=0)
        self.variance = np.var(input, axis=0)
        
        self.normalized = (self.input - self.mean) / np.sqrt((self.variance + epsilon))
        
        self.out = np.multiply(self.normalized, self.gamma) + self.beta
        return self.out


    def backward(self, dLdZ, step):
        self.dLdX = np.zeros(self.input.shape)
        self.dLdG = np.sum(np.multiply(self.normalized, dLdZ), axis=0)
        self.dLdB = np.sum(dLdZ, axis=0)

        self.dXhatdVar = -(self.input - self.mean) / (2 * (self.variance + self.epsilon)**(3/2))
        self.dXhatdMu = (-1. / np.sqrt(self.variance + self.epsilon))
        self.dLdXhat = np.multiply(dLdZ, self.gamma)

        #print("mean", self.mean[:2])
        #print("epsilon", self.epsilon)
        #print("dLdVar", self.dXhatdVar[:2])
        #print("dLdMu", self.dXhatdMu[:2])
        #print("dLdXhat", self.dLdXhat[:2])
        self.dLdX = self.dLdXhat * ((1 / np.sqrt(self.variance + self.epsilon)) + self.dXhatdVar * ((2. * (self.input - self.mean)) / np.prod(self.mean.shape)) + self.dXhatdMu * (1. / np.prod(self.mean.shape)))

        return self.dLdX


