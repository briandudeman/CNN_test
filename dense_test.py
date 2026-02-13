import numpy as np
import dense
import relu
import loss

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math


#----------------------------------------------Preprocessing data-------------------------------------------------------

data = fetch_california_housing()
X, Y = data["data"], data["target"]
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2)
'''
ptratio = X_train[:, 10]
ptratio_Xtest = X_test[:, 10]
lstat = X_train[:, 12]
lstat_Xtest = X_test[:, 12]
'''
y_train_shaped = np.reshape(Y_train, (Y_train.shape[0], 1))
#-----------------------------------------------------------------------------------------------------------------------


#-------------------------------------------------Declaring Network------------------------------------------------------

network = [
    dense.FC_layer(13, 5),
    relu.Relu(),
    dense.FC_layer(5, 5),
    relu.Relu(),
    dense.FC_layer(5, 1),
]

final_loss = loss.Mse()

e = 100
losses = []
epochs = []
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------Functions---------------------------------------------------------


def predict(network, input, epoch):
    output = input
    layer_num = 0
    for layer in network:
        output = layer.forward(output, epoch, layer_num, .03)
        layer_num += 1
    return output


def standardize(data):  # for standardizing either the input or output of the network. not used in the current model
    mean = np.sum(data)/np.size(data)
    sd = math.sqrt(np.sum(np.square(data-mean))/np.size(data))
    return (data-mean)/sd
#-----------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------Training Model-------------------------------------------------------

for i in range(e):
    error = 0
    epochs.append(i)

    output = predict(network, X_train.T, i) # forward propagation

    error += final_loss.mse(output, Y_train.T) # error computation, currently not used
    losses.append(final_loss.root_mse(output, Y_train.T))

    # backpropagation
    grad = final_loss.mse_prime(output, Y_train.T)
    layer_num = len(network)
    for layer in reversed(network):
        grad = layer.backward(grad, i, layer_num, .03)
        layer_num = layer_num - 1

#-----------------------------------------------------------------------------------------------------------------------

#----------------------------------------------Plotting Data----------------------------------------------------------
prediction = predict(network, X_train.T, None)# getting final prediction
training_loss = final_loss.root_mse(prediction, Y_train.T)
prediction_test = predict(network, X_test.T, None)
test_loss = final_loss.root_mse(prediction_test, Y_test.T)
print("training loss", training_loss)
print("test loss", test_loss)

plt.figure()
plt.title("lstat training data(blue) vs prediction(red)")
plt.xlabel("lstat")
plt.ylabel("training/prediction")
plt.scatter(lstat, prediction, c="Red")  # predictions related to one of the parameters of the house. it is in red
plt.scatter(lstat, Y_train)  # actual

plt.figure()
plt.title("losses over time")
plt.xlabel("epochs")
plt.ylabel("losses")
plt.plot(epochs, losses)  # losses

plt.figure()
plt.title("Ptratio training data(blue) vs prediction(red)")
plt.xlabel("ptratio")
plt.ylabel("training/prediction")
plt.scatter(ptratio, Y_train)  # actual
plt.scatter(ptratio, prediction, c='Red')  # predictions against another parameter. it is in red

plt.show()

#-----------------------------------------------------------------------------------------------------------------------