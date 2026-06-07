import numpy as np


from layers.python_layers.layer import Layer


class MaxPoolingLayer(Layer):

  def __init__(self, kernel_size: int, stride = None):
    self.kernel_size = kernel_size # defining number of rows and columns in a filter

    self.stride = kernel_size
    if (stride):
      self.stride = stride
    super().__init__()
    

  def forward(self, input):
    self.input = input
    self.p, self.d, self.m, self.n = input.shape # defining the depth and number of rows and columns of the input. The depth of the filters is d as well
    
    self.out_m = int((self.m - self.kernel_size)/self.stride + 1) # using formula to get the size of the output
    self.out_n = int((self.n - self.kernel_size)/self.stride + 1)
    self.out = np.zeros((self.p, self.d, self.out_m, self.out_n))
    #print(self.out.shape)
    for p in range(self.p):
      for i in range(self.d):
        for j in range(self.out_m): # subtracting kernel dimension to keep in bounds
          for k in range(self.out_n):
            #print((i, j, k))
            self.out[p, i, j, k] = np.amax(input[p, i, (j * self.stride):(j * self.stride + self.kernel_size), (k * self.stride):(k * self.stride + self.kernel_size)])

    return self.out


  def backward(self, dLdO):
    self.dLdO = dLdO
    self.dLdX = np.zeros(self.input.shape)

    for p in range(self.p):
      for i in range(self.d):
        for j in range(self.out_m): # subtracting kernel dimension to keep in bounds
          for k in range(self.out_n):
            max_index = np.where(self.input==np.amax(self.input[p, i, (j * self.stride):(j * self.stride + self.kernel_size), (k * self.stride):(k * self.stride + self.kernel_size)]))
            self.dLdX[max_index[0], max_index[1], max_index[2], max_index[3]] = dLdO[p, i, j, k] #setting indexes in input gradient to the derivatives of input

    return self.dLdX




