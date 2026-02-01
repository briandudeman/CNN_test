import numpy as np

X = np.array([[[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]]])

class MaxPoolingLayer:

  def __init__(self, input_size: tuple[int, ...], kernel_size: int, stride = None):
    self.d, self.m, self.n = input_size # defining the depth and number of rows and columns of the input. The depth of the filters is d as well
    self.kernel_size = kernel_size # defining number of rows and columns in a filter

    self.stride = kernel_size
    if (stride):
      self.stride = stride
    
    self.out_m = int((self.m - self.kernel_size)/self.stride + 1) # using formula to get the size of the output
    self.out_n = int((self.n - self.kernel_size)/self.stride + 1)
    self.out = np.zeros((self.d, self.out_m, self.out_n))
    print(self.out.shape)

  def forward(self, input):
    self.input = input
    #TODO: extra for loop for d 
    for i in range(self.d):
      for j in range(self.out_m): # subtracting kernel dimension to keep in bounds
        for k in range(self.out_n):
          print((i, j, k))
          self.out[i, j, k] = np.amax(input[i, (j * self.stride):(j * self.stride + self.kernel_size), (k * self.stride):(k * self.stride + self.kernel_size)])

    return self.out


  def backward(self, dLdO):
    self.dLdO = dLdO
    self.dLdX = np.zeros((self.d, self.m, self.n))

    for i in range(self.d):
      for j in range(self.out_m): # subtracting kernel dimension to keep in bounds
        for k in range(self.out_n):
          max_index = np.where(self.input==np.amax(self.input[i, (j * self.stride):(j * self.stride + self.kernel_size), (k * self.stride):(k * self.stride + self.kernel_size)]))
          self.dLdX[max_index[0], max_index[1], max_index[2]] = dLdO[i, j, k] #setting indexes in input gradient to the derivatives of input

    return self.dLdX




