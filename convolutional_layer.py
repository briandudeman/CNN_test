import numpy as np
from scipy import signal
import copy

# each filter contains d many kernels
# using this as a guide: https://learnopencv.com/understanding-convolutional-neural-networks-cnn/

np.random.seed(6)


class Conv_Layer:
  def __init__(self, input_size, kernel_size, filter_num, stride = 1, padding = None):
    self.d, self.m, self.n = input_size # defining the depth and number of rows and columns of the input. The depth of the filters is d as well
    self.f_num = filter_num
    self.k_m, self.k_n = kernel_size # defining number of rows and columns in a filter
    
    if (padding == None): # assuming the kernel size is odd :/
      padding = (int)(self.k_n / 2)
    self.padding = padding
    
    self.stride = stride

    self.out_m = ((self.m + 2 * self.padding - self.k_m) / self.stride) + 1 # using formula to get the size of the output
    self.out_n = ((self.n + 2 * self.padding - self.k_n) / self.stride) + 1 
    
    
    self.out = np.zeros((self.f_num, self.out_m, self.out_n))
    self.filters = np.random.randn(self.f_num, self.d, self.k_m, self.k_n) # randomizing filters and biases
    self.bias = np.random.randn(self.f_num, self.out_m, self.out_n)
    print(self.bias)
    #print("bias", self.bias, self.f_num, self.out_m, self.out_n)
    #print("filters", self.filters, np.shape(self.filters))

  def forward(self, input: np.ndarray):

    self.input = input

    # padding the input
    padded_input = np.zeros(self.d, self.m + self.padding, self.n + self.padding)
    self.padded_input = padded_input[:, (self.padding):(self.padding + self.m + 1), (self.padding):(self.padding + self.n + 1)] = input


    self.out = copy.deepcopy(self.bias) # makes adding everything with the bias easier
    for i in range(self.f_num): # for each filter or each output
      for j in range(self.d):
        self.out[i] += self.convolve(padded_input[j], self.filters[i, j], self.stride)
    return self.out

  # (2D) input is an m * n array, filter is somthing like k_m * k_n
  def convolve(self, input: np.ndarray, filter: np.ndarray, stride: int, backprop: bool = False):
    convolve_out = np.zeros(input.shape) # no support for higher dimensions, shouldn't need to be any
    
    for i in range(0, input.shape[0] - filter.shape[0], stride): # subtracting kernel dimension to keep in bounds
      for j in range(0, input.shape[1] - filter.shape[0], stride):
        convolve_out[i, j] = np.sum(input[(i):(i + filter.shape[0]), (j):(j + filter.shape[0])] * filter)

    return convolve_out


  def backward(self, dLdO, l_rate=.01): # should return the partial derivative of the loss with respect to the input to the layer, kernels/filters, and the biases(not using automatic differentiation)
    #print("shape", np.shape(dLdZ))
    self.dLdF = np.zeros(np.shape(self.filters)) #gradient of loss with respect to filters/kernels of layer
    self.dLdB = copy.deepcopy(dLdO) #gradient of loss with respect to the biases
    self.dLdX = np.zeros(np.shape(self.input)) # gradient with respect to the input of the layer
    
    padded_dLdO = np.zeros(dLdO.shape[0], dLdO.shape[1] + self.padding, dLdO.shape[2] + self.padding)
    padded_dLdO = padded_dLdO[:, (self.padding):(self.padding + dLdO.shape[1] + 1), (self.padding):(self.padding + dLdO.shape[2] + 1)] = dLdO
    
    
    for i in range(self.f_num):
      for j in range(self.d):
        self.dLdX[j] += self.convolve(padded_dLdO[i], np.rot90(self.filters[i, j], 2, (0, 1)), self.stride) # TODO: the dimensions on this arent right, how to convolve 1 x m x n and d x m x n?
      self.dLdF[i] = self.convolve(self.padded_input, dLdO, self.stride)

    self.filters -= l_rate * self.dLdF
    self.bias -= l_rate * self.dLdB
    #print("dLdF: ", self.dLdF, "dLdB: ", self.dLdB, "dLdX: ", self.dLdX)
    return self.dLdX


