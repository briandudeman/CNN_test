

class BaseOptimizer:
    def __init__(self, lr):
        self.lr = lr

    def backward(self, grad, epoch):
        pass
