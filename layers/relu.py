import numpy as np

from layers.layer import Layer

class ReLu(Layer):

  def forward(self, z):
    self.z = z
    return np.where(self.z >= 0, self.z, -0.1*self.z) # leaky relu

  def backward(self, dLdA, epoch, lr=0.01):
    self.dLdA = dLdA
    return np.where(dLdA >= 0, dLdA, -0.1*dLdA)




