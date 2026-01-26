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

  def forward(self, input):
    self.input = input
    #print("shape", np.shape(input))
    self.out = copy.deepcopy(self.bias) # makes adding everything with the bias easier
    for i in range(self.f_num): # for each filter or each output
      for j in range(self.d): # for each layer of the input
        self.out[i] += signal.correlate2d(self.input[j], self.filters[i, j], "valid")
    return self.out

  def backward(self, dLdZ, l_rate=.01): # should return the partial derivative of the loss with respect to the input to the layer, kernels/filters, and the biases(not using automatic differentiation)
    self.dLdZ = dLdZ
    self.dLdz_d, self.dLdZ_m, self.dLdZ_n = np.shape(dLdZ)
    #print("shape", np.shape(dLdZ))
    self.dLdF = np.zeros(np.shape(self.filters)) #gradient of loss with respect to filters/kernels of layer
    self.dLdB = copy.deepcopy(self.dLdZ) #gradient of loss with respect to the biases
    self.dLdX = np.zeros((self.d, self.m, self.n)) # gradient with respect to the input of the layer
    for i in range(self.f_num):
      for j in range(self.d):
        print(np.shape(signal.convolve2d(self.dLdZ[i], self.filters[i, j], "full")))
        print(np.shape(self.filters[i, j]))
        self.dLdX[j] += signal.convolve2d(self.dLdZ[i], self.filters[i, j], "full")
        self.dLdF[i, j] = signal.correlate2d(self.input[j], self.dLdZ[i], "valid") #correlate is the filter regular, convolve is with it flipped 180 degrees

    self.filters -= l_rate * self.dLdF
    self.bias -= l_rate * self.dLdB
    #print("dLdF: ", self.dLdF, "dLdB: ", self.dLdB, "dLdX: ", self.dLdX)
    return self.dLdX


