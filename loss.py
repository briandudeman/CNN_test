import numpy as np


class Loss:
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


