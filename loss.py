import numpy as np
import math


class CrossEntropy:
    def __init__(self, y):
        self.y = y.reshape((y.shape[0], y.shape[1], 1)) # for batching

    def cross_entropy(self, y_pred):
        self.y_pred = y_pred
        #print(self.y_pred[0])
        #print(np.log10(self.y_pred[0]))
        return -np.sum(self.y * np.log10(self.y_pred)) / self.y.shape[0] # divide by batch size for error

    def cross_entropy_derivative(self):
        dLdY = self.y_pred - self.y
        return dLdY
    
class Mse:

    def root_mse(self, y_pred, y):
        #print("y pred shape:", y_pred.shape)
        #print("y actual shape:", y.shape)
        #print(math.sqrt(np.mean(np.sum((y - y_pred) ** 2, 1))))
        return math.sqrt(np.mean(np.sum((y - y_pred) ** 2, 1)))

    def mse(self, y_pred, y):
        return np.mean((1 / y.shape[1] * np.sum(np.square(y_pred - y), 1)))

    def mse_prime(self, y_pred, y):
        #print("y pred shape prime:", y_pred.shape)
        #print("y actual shape prime:", y.shape)
        return (2 * (y_pred - y) / np.prod(y.shape))


