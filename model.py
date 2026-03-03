

from layer import Layer
from optimizers.base_optimizer import BaseOptimizer


class Model:
    def __init__(self, layers: list[Layer], optimizer: BaseOptimizer):
        self.layers = layers
        self.optimizer = optimizer
    
    def predict(self, input):
        output = input
        for i, layer in enumerate(self.layers):
            #print("output shape ", i, " ", output.shape)
            output = layer.forward(output)
        return output

    def backprop(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr=self.optimizer.lr)
        
