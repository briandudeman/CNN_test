import numpy as np
import seaborn as sns
import pandas as pd

import dense
import relu
import loss
from batch_normalization import BatchNormalization

import sklearn.datasets as dt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math

#----------------------------------------------Preprocessing data-------------------------------------------------------

data = fetch_california_housing()
california_housing = fetch_california_housing(as_frame=True)
X, Y = data["data"], data["target"]
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2)

med_income_train = X_train[:, 0]
med_income_test = X_test[:, 0]
house_age_train = X_train[:, 1]
house_age_test = X_test[:, 1]

y_train_shaped = np.reshape(Y_train, (Y_train.shape[0], 1))
#-----------------------------------------------------------------------------------------------------------------------


#-------------------------------------------------Declaring Network------------------------------------------------------

network = [
    dense.FCLayer(50),
    relu.ReLu(),
    dense.FCLayer(100),
    relu.ReLu(),
    dense.FCLayer(100),
    relu.ReLu(),
    dense.FCLayer(1),
]

final_loss = loss.Mse()

e = 100
losses = []
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------Functions---------------------------------------------------------



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

mini_batches = make_mini_batches(X_train, y_train_shaped, 1)


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def standardize(data):  # for standardizing either the input or output of the network. not used in the current model
    mean = np.sum(data)/np.size(data)
    sd = math.sqrt(np.sum(np.square(data-mean))/np.size(data))
    return (data-mean)/sd
#-----------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------Training Model-------------------------------------------------------

for i in range(len(mini_batches)):
    print("epoch: ", i)
    error = 0
    x_train_batch = np.array(mini_batches[i][0]).reshape(np.array(mini_batches[i][0]).shape + (1, ))
    y_train_batch = np.array(mini_batches[i][1]).reshape(np.array(mini_batches[i][1]).shape + (1, ))
    #print("x batch shape: ", x_train_batch.shape)
    #print("y batch shape: ", y_train_batch.shape)
    print("x_batch standardized: ", standardize(x_train_batch))
    output = predict(network, standardize(x_train_batch)) # forward propagation
    print("output", output)
    print("y actual", y_train_batch)
    error += final_loss.mse(output, y_train_batch) # error computation, currently not used
    losses.append(final_loss.root_mse(output, y_train_batch))

    # backpropagation
    grad = final_loss.mse_prime(output, y_train_batch)
    print("grad: ", grad)
    for layer in reversed(network):
        grad = layer.backward(grad)

#-----------------------------------------------------------------------------------------------------------------------

#----------------------------------------------Plotting Data----------------------------------------------------------
prediction = predict(network, standardize(X_train.reshape(X_train.shape + (1, ))))# getting final prediction
training_loss = final_loss.root_mse(prediction, Y_train.reshape(Y_train.shape + (1,1)))
prediction_test = predict(network, standardize(X_test.reshape(X_test.shape + (1, ))))
test_loss = final_loss.root_mse(prediction_test, Y_test.reshape(Y_test.shape + (1,1)))
print("training loss", training_loss)
print("test loss", test_loss)
print(prediction[:5])
print(Y_train[:5])

plt.figure()
plt.title("losses over time")
plt.xlabel("epochs")
plt.ylabel("losses")
plt.plot(losses)  # losses

plt.figure()
scatter_data = np.vstack((X_train[:, 6], X_train[:, 7]))
scatter_data = np.vstack((prediction.reshape((prediction.shape[0], prediction.shape[1])).T, scatter_data))
scatter_data = pd.DataFrame(scatter_data.T, index=None, columns=["preds", "Latitude", "Longitude"])

sns.scatterplot(
    data=scatter_data,
    x="Longitude",
    y="Latitude",
    hue="preds",
    palette="viridis",
    alpha=0.5,
)
plt.legend(title="MedHouseVal", bbox_to_anchor=(1.05, 0.95), loc="upper left")
_ = plt.title("Median house value depending of\n their spatial location")

plt.show()

#-----------------------------------------------------------------------------------------------------------------------
