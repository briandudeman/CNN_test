import numpy as np

from optimizers.base_optimizer import BaseOptimizer

class Adam(BaseOptimizer):
    def __init__(self, lr, beta_one=0.9, beta_two=0.999, epsilon=1e-8):
        super().__init__(lr)
        self.beta_one = beta_one
        self.beta_two = beta_two
        self.epsilon = epsilon
        self.m = None
        self.v = None

    def backward(self, grad, epoch):
        #print(grad.shape)
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)

        self.m = self.beta_one * self.m + (1 - self.beta_one) * grad
        self.v = self.beta_two * self.v + (1 - self.beta_two) * np.square(grad)

        m_hat = self.m / (1 - self.beta_one ** epoch)
        v_hat = self.v / (1 - self.beta_two ** epoch)
        #print(self.lr * (m_hat / (np.sqrt(v_hat) + self.epsilon)))
        return self.lr * (m_hat / (np.sqrt(v_hat) + self.epsilon))

