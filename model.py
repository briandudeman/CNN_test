

from layers.python_layers.layer import Layer
from layers.python_layers.weight_layer import WeightLayer
from optimizers.python_optimizers.base_optimizer import BaseOptimizer


import copy


class Model:
    def __init__(self, layers: list[Layer], optimizer: BaseOptimizer, *optimizer_args):
        self.layers = layers
        self.optimizer_type = optimizer
        for layer in self.layers:
            if isinstance(layer, WeightLayer):
                layer.set_optimizer(self.optimizer_type, *optimizer_args)
    
    def predict(self, input):
        output = input
        for i, layer in enumerate(self.layers):
            #print("output shape ", i, " ", output.shape)
            output = layer.forward(output)
        return output

    def backprop(self, grad, epoch):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, epoch, lr=layer.lr)
        
