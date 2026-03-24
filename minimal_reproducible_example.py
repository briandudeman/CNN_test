from model import Model
import numpy as np
import seaborn as sns
import pandas as pd

import layers.relu as relu
import loss


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math

class Layer:
    def __init__(self):
        self.lr = .0001

    def forward(self):
        pass
    

    def backward(self):
        pass

class ReLu(Layer):

  def forward(self, z):
    self.z = z
    return np.where(self.z >= 0, self.z, -0.1*self.z) # leaky relu

  def backward(self, dLdA, epoch, lr=0.01):
    self.dLdA = dLdA
    return np.where(dLdA >= 0, dLdA, -0.1*dLdA)

class Mse:

    def root_mse(self, y_pred, y):
        #print("y pred shape:", y_pred.shape)
        #print("y actual shape:", y.shape)
        #print(math.sqrt(np.mean(np.sum((y - y_pred) ** 2, 1))))
        return math.sqrt(np.mean(np.sum((y - y_pred) ** 2, 1)))

    def mse(self, y_pred, y):
        return np.mean((1 / y.shape[1] * np.sum(np.square(y_pred - y), 1)))

    def mse_prime(self, y_pred, y):
        #print("y pred shape prime:", y_pred.shape)
        #print("y actual shape prime:", y.shape)
        return (2 * (y_pred - y) / np.prod(y.shape))


class WeightLayer(Layer):
    def __init__(self):
        super().__init__()

    def set_optimizer(self, optimizer, *optimizer_args):
        self.optimizer_weights = optimizer(*optimizer_args)
        self.optimizer_biases = optimizer(*optimizer_args)

class FCLayer(WeightLayer):
    def __init__(self, num_nodes):
      # being fed through(batch size)
      self.m = num_nodes  # number of nodes
      self.called_first_time = False
      super().__init__()

    def forward(self, input):
      self.input = input
      #print("input shape", input.shape)
      self.p, self.d, self.n = input.shape  # p is batch size, d is number of dimensions to the data and n is the number of data points

      if not self.called_first_time:
        self.called_first_time = True
        self.x_limit = math.sqrt((2 / (self.n + self.m)))
        self.weights = np.random.uniform(-self.x_limit, self.x_limit, size=(self.m, self.d))  # d by m
        self.biases = np.random.uniform(-self.x_limit, self.x_limit, size=(self.m, 1))

      #print("weights", self.weights)
      #print("biases", self.biases)
      self.z = np.zeros((self.p, self.m, self.n))
      for p in range(self.p):
        self.z[p] = (self.weights @ self.input[p]) + self.biases
      return self.z
    
    def backward(self, dLdZ, epoch, lr=.00001):
      #print("dLdZ shape", dLdZ.shape)
      self.dLdA = self.weights.T @ dLdZ

      self.dLdW = np.array([dLdZ[i] @ self.input[i].T for i in range(self.input.shape[0])])
      self.dLdW = np.sum(self.dLdW, axis=0)
      # gradients for biases should sum over batch and feature map dims, result shape (m,1)
      self.dLdW0 = np.sum(dLdZ, axis=0)
      # Clip gradients to prevent explosion
      self.dLdW = np.clip(self.dLdW, -1.0, 1.0)
      self.dLdW0 = np.clip(self.dLdW0, -1.0, 1.0)
      #print("dldw", self.dLdW)
      #print("shape of dLdW0, in linear", np.shape(self.dLdW0))
      #print("shape of dLdW, in linear", np.shape(self.dLdW))
      self.weights -= self.optimizer_weights.backward(self.dLdW, epoch)
      self.biases -= self.optimizer_biases.backward(self.dLdW0, epoch)
      return self.dLdA  # d by n


class Adam():
    def __init__(self, lr, beta_one=0.9, beta_two=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta_one = beta_one
        self.beta_two = beta_two
        self.epsilon = epsilon
        self.m = None
        self.v = None

    def backward(self, grad, epoch):
        #print(grad.shape)
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)

        self.m = self.beta_one * self.m + (1 - self.beta_one) * grad
        self.v = self.beta_two * self.v + (1 - self.beta_two) * np.square(grad)

        m_hat = self.m / (1 - self.beta_one ** epoch)
        v_hat = self.v / (1 - self.beta_two ** epoch)
        #print(self.lr * (m_hat / (np.sqrt(v_hat) + self.epsilon)))
        return self.lr * (m_hat / (np.sqrt(v_hat) + self.epsilon))



class Model:
    def __init__(self, layers: list[Layer], optimizer: Adam, *optimizer_args):
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
        


def standardize(data, mean = None, std = None):
    if not isinstance(mean, np.ndarray):
        mean = np.average(data, axis=0)
    if not isinstance(std, np.ndarray):
        std = np.std(data, axis=0)
    return (data - mean) / std

def destandardize(data, mean = None, std = None):
    if not isinstance(mean, np.ndarray):
        mean = np.average(data, axis=0)
    if not isinstance(std, np.ndarray):
        std = np.std(data, axis=0)
    return data * std + mean


def make_mini_batches(x, y, batch_size):
    mini_batches = []

    data = [(x[i], y[i]) for i in range(x.shape[0])]
    np.random.shuffle(data)
    for i in range((len(data) // batch_size)):
        mini_batch = data[i * batch_size:(i + 1) * batch_size]
        X_mini = np.array([mini_batch[j][0] for j in range(len(mini_batch))])

        Y_mini = np.array([mini_batch[j][1] for j in range(len(mini_batch))])

        mini_batches.append((X_mini, Y_mini))
    return mini_batches


#----------------------------------------------Preprocessing data-------------------------------------------------------

data = fetch_california_housing()
california_housing = fetch_california_housing(as_frame=True)
X, Y = data["data"], data["target"]
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2)

x_train_mean = X_train.mean(axis=0)
x_train_mean = x_train_mean.reshape((x_train_mean.shape + (1,))).reshape((1, -1))

x_train_std = X_train.std(axis=0) + 1e-8
x_train_std = x_train_std.reshape((x_train_std.shape + (1,))).reshape((1, -1))

y_train_mean = Y_train.mean(axis=0)
y_train_mean = y_train_mean.reshape((y_train_mean.shape + (1,))).reshape((1, -1))

y_train_std = Y_train.std(axis=0) + 1e-8
y_train_std = y_train_std.reshape((y_train_std.shape + (1,))).reshape((1, -1))

learning_rate = 1e-6  # adjust downward if loss starts climbing after a while

med_income_train = X_train[:, 0]
med_income_test = X_test[:, 0]
house_age_train = X_train[:, 1]
house_age_test = X_test[:, 1]

Y_train = np.reshape(Y_train, (Y_train.shape[0], 1))
#-----------------------------------------------------------------------------------------------------------------------


#-------------------------------------------------Declaring Network------------------------------------------------------

network = Model([
    FCLayer(50),
    ReLu(),
    FCLayer(100),
    ReLu(),
    FCLayer(100),
    ReLu(),
    FCLayer(1),
], Adam, 1e-5)

final_loss = Mse()

epochs = 1000
mini_batch_size = 32
losses = []

#--------------------------------------------------Preprocessing Data-------------------------------------------------------

X_train = standardize(X_train, x_train_mean, x_train_std).reshape((X_train.shape + (1,)))
Y_train = standardize(Y_train, y_train_mean, y_train_std).reshape((Y_train.shape + (1,)))

X_test = standardize(X_test, x_train_mean, x_train_std).reshape((X_test.shape + (1,)))
Y_test = Y_test.reshape(Y_test.shape + (1,1))
#-----------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------Training Model-------------------------------------------------------
for e in range(epochs):
    # shuffle / rebuild mini‑batches each epoch
    mini_batches = make_mini_batches(X_train, Y_train, mini_batch_size)
    epoch_losses = []
    print(f"epoch {e+1}/{epochs}")

    for x_batch, y_batch in mini_batches:
        # batches already standardized in make_mini_batches
        output = network.predict(x_batch)
        batch_loss = final_loss.mse(output, y_batch)
        epoch_losses.append(batch_loss)

        grad = final_loss.mse_prime(output, y_batch)
        network.backprop(grad, e+1)

    avg = np.mean(epoch_losses)
    losses.append(avg)
    if (e+1) % 10 == 0 or e == 0:
        print(f"  avg mse {avg:.4f}")

#-----------------------------------------------------------------------------------------------------------------------

#----------------------------------------------Plotting Data----------------------------------------------------------
# evaluate on original scale
prediction = network.predict(X_train)
prediction = destandardize(prediction, y_train_mean, y_train_std)
training_loss = final_loss.root_mse(prediction, Y_train)

prediction_test = network.predict(X_test)
prediction_test = destandardize(prediction_test, y_train_mean, y_train_std)
test_loss = final_loss.root_mse(prediction_test, Y_test)

print("training loss", training_loss)
print("test loss", test_loss)
print("avg pred", prediction.mean(), "avg target", Y_train.mean())

plt.figure()
plt.title("losses over time")
plt.xlabel("epochs")
plt.ylabel("losses")
plt.plot(losses)  # losses

plt.figure()
