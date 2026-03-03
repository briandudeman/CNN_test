import numpy as np

from base_optimizer import BaseOptimizer

class Adam(BaseOptimizer):
    def __init__(self, lr):
        self.lr = lr
        self.first_pass = True
        self.epsilon = .00000001

    def backward(self, grad, epoch):

        if self.first_pass:
            self.beta_one = .9
            self.beta_two = .9
            self.m = np.zeros(grad.shape)
            self.v = np.zeros(grad.shape)
            self.first_pass = False

        self.m = self.beta_one * self.m + (1 - self.beta_one) * grad
        self.v = self.beta_two * self.v + (1 - self.beta_two) * np.square(grad)

        m_hat = self.m / (1 - self.beta_one ** epoch)
        n_hat = self.m / (1 - self.beta_two ** epoch)

        return (m_hat / np.sqrt(n_hat + self.epsilon)) * self.lr

