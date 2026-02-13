import numpy as np
import copy
import scipy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

from convolutional_layer import ConvLayer
from max_pooling_layer import MaxPoolingLayer
from max_pooling_layer import MaxPoolingLayer
from relu import ReLu
from reshape import Reshape
from dense import FCLayer
from softmax import SoftMax
from loss import Loss
from batch_normalization import BatchNormalization



(x_train, y_train), (x_test, y_test) = mnist.load_data()

def process(x, y, limit, num_classes):
    x = x[0:limit][:][:]
    y = y[0:limit][:][:]

    x = x.astype("float32") / 255
    x = np.expand_dims(x, 1)
    y = np.expand_dims(y, 1)

    y = to_categorical(y, num_classes)
    return x, y


x_train, y_train = process(x_train, y_train, 20000, 10)
x_test, y_test = process(x_test, y_test, 10000, 10)

def make_mini_batches(x, y, batch_size):
    mini_batches = []
    data = [(x[i], y[i]) for i in range(x.shape[0])]
    np.random.shuffle(data)
    for i in range((len(data) // batch_size)):
        mini_batch = data[i * batch_size:(i + 1) * batch_size]
        X_mini = [mini_batch[j][0] for j in range(len(mini_batch))]
        Y_mini = [mini_batch[j][1] for j in range(len(mini_batch))]
        mini_batches.append((X_mini, Y_mini))
    return mini_batches

mini_batches = make_mini_batches(x_train, y_train, 32)
'''
for batch in mini_batches:
    print(np.array(batch[0]).shape)
    print(np.array(batch[1]).shape)
print(len(mini_batches))
'''
layers = [ConvLayer(5, 16),
          BatchNormalization(),
          ReLu(),
          MaxPoolingLayer(2),
          ConvLayer(3, 10),
          ReLu(),
          MaxPoolingLayer(2),
          BatchNormalization(),
          Reshape(),
          FCLayer(50),
          ReLu(),
          BatchNormalization(),
          FCLayer(10),
          SoftMax()]


losses = []

for i in range(len(mini_batches)):    
    input = np.array(mini_batches[i][0])
    for j, layer in enumerate(layers):
        #print("layer ", j, " ", layer, " shape ", input.shape)
        input = layer.forward(input)
    
    #print("ytrain: ", y_train[i].shape)
    #print("y: ", input.shape)
    loss = Loss(np.array(mini_batches[i][1]))

    #print("error", loss.cross_entropy(input))
    losses.append(loss.cross_entropy(input))
    dLdY = loss.cross_entropy_derivative()

    for j, layer in enumerate(layers[::-1]):
        #print("layer ", len(layers) - j - 1, " shape ", dLdY.shape)
        dLdY = layer.backward(dLdY)

guesses = []
actuals = []
right_guesses = 0
for i in range(x_test.shape[0]):    
    input = np.reshape(x_test[i, :, :, :], (1, x_test[i, :, :, :].shape[0], x_test[i, :, :, :].shape[1], x_test[i, :, :, :].shape[2]))
    #print(input.shape)
    for j, layer in enumerate(layers):
        #print("layer ", j, " shape ", input.shape)
        input = layer.forward(input)
    
    
    guesses.append(np.argmax(input))
    actuals.append(np.argmax(y_test[i]))
    if (guesses[i] == actuals[i]):
        right_guesses += 1
    #print("ytrain: ", y_train[i].shape)
    #print("y: ", input.shape)
    #print(y_test[i].shape)
    loss = Loss(np.reshape(y_test[i], y_test[i].shape + (1,)))

    #print("error", loss.cross_entropy(input))
    epoch_loss = loss.cross_entropy(input)

print("acc: ", right_guesses / x_test.shape[0])

plt.plot(losses)
plt.show()
