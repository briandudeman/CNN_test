import numpy as np
import copy

# each filter contains d many kernels
# using this as a guide: https://learnopencv.com/understanding-convolutional-neural-networks-cnn/

np.random.seed(6)


class ConvLayer:
  def __init__(self, input_shape: tuple[int, ...], kernel_size: int, filter_num: int, stride = 1, padding = None):
    
    if kernel_size % 2 == 0:
      raise NotImplementedError("Even sized kernels have not been implemented due to asymmetries, please change your kernel size.")
    
    self.d, self.m, self.n = input_shape # defining the depth and number of rows and columns of the input. The depth of the filters is d as well
    self.f_num = filter_num
    self.kernel_size = kernel_size

    if (padding == None): # assuming the kernel size is odd :/
      padding = (int)(self.kernel_size / 2)
    self.padding = padding
    
    self.stride = stride

    self.out_m = (int)((self.m + 2 * self.padding - self.kernel_size) / self.stride) + 1 # using formula to get the size of the output
    self.out_n = (int)((self.n + 2 * self.padding - self.kernel_size) / self.stride) + 1 
    
    self.out = np.zeros((self.f_num, self.out_m, self.out_n))
    self.filters = np.random.randn(self.f_num, self.d, self.kernel_size, self.kernel_size) # randomizing filters and biases
    self.bias = np.random.randn(self.f_num, self.out_m, self.out_n)
    #print("bias", self.bias, self.f_num, self.out_m, self.out_n)
    #print("filters", self.filters, np.shape(self.filters))

  def forward(self, input: np.ndarray):

    self.input = input

    self.out = copy.deepcopy(self.bias) # makes adding everything with the bias easier
    for i in range(self.f_num): # for each filter or each output
      for j in range(self.d):
        self.out[i] += self.convolve(input[j], self.filters[i, j], self.stride)
    return self.out

  # (2D) input is an m * n array, filter is somthing like kernel_size^2
  def convolve(self, input: np.ndarray, filter: np.ndarray, stride: int, backprop: bool = False):
    
    # padding the input
    padded_input = np.zeros((input.shape[0] + (2 * self.padding), input.shape[1] + (2 * self.padding)))
    padded_input[(self.padding):(self.padding + input.shape[0]), (self.padding):(self.padding + input.shape[1])] = input

    
    cout_m = (int)((input.shape[0] + 2 * self.padding - filter.shape[0]) / self.stride) + 1 # using formula to get the size of the output
    cout_n = (int)((input.shape[1] + 2 * self.padding - filter.shape[1]) / self.stride) + 1 
    convolve_out = np.zeros((cout_m, cout_n)) # no support for higher dimensions, shouldn't need to be any
    
    for i in range(0, input.shape[0] - filter.shape[0], stride): # subtracting kernel dimension to keep in bounds
      for j in range(0, input.shape[1] - filter.shape[0], stride):
        convolve_out[i, j] = np.sum(input[(i):(i + filter.shape[0]), (j):(j + filter.shape[0])] * filter)

    return convolve_out


  def backward(self, dLdO, l_rate=.01): # should return the partial derivative of the loss with respect to the input to the layer, kernels/filters, and the biases(not using automatic differentiation)
    #print("shape", np.shape(dLdZ))
    self.dLdF = np.zeros(np.shape(self.filters)) #gradient of loss with respect to filters/kernels of layer
    self.dLdB = copy.deepcopy(dLdO) #gradient of loss with respect to the biases
    self.dLdX = np.zeros(np.shape(self.input)) # gradient with respect to the input of the layer
    
    
    for i in range(self.f_num):
      for j in range(self.d):
        self.dLdX[j] += self.convolve(dLdO[i], np.rot90(self.filters[i, j], 2, (0, 1)), self.stride)
        self.dLdF[i, j] = self.convolve(self.input[j], dLdO[i], self.stride)

    self.filters -= l_rate * self.dLdF
    self.bias -= l_rate * self.dLdB
    #print("dLdF: ", self.dLdF, "dLdB: ", self.dLdB, "dLdX: ", self.dLdX)
    return self.dLdX


