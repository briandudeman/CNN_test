import numpy as np
import math

from layers.python_layers.layer import Layer

class Reshape(Layer): # reshaping for fc layers

    def forward(self, input):
        self.input_shape = input.shape
        batch_size = input.shape[0]
        input_shape_batch_accounted = input.shape[1: ]
        return np.reshape(input, (batch_size, math.prod(input_shape_batch_accounted), 1))

    def backward(self, output):
        return np.reshape(output, self.input_shape)



