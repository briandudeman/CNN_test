import numpy as np
from convolutional_layer import ConvLayer
from max_pooling_layer import MaxPoolingLayer

# depth, height, width
input_shape = (3, 10, 11)
output_shape = (3, 8, 9)
kernel_size = 3
input = np.random.rand(input_shape[0], input_shape[1], input_shape[2])

layer = MaxPoolingLayer(input_shape, kernel_size, stride = 1)

output = layer.forward(input)

print("output is correct size: ", output.shape == output_shape)
print(output.shape)
print(output_shape)

input_err = layer.backward(output)

print("input error is correct size: ", input_err.shape == input_shape)
