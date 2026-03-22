from model import Model
import numpy as np
import seaborn as sns
import pandas as pd

import layers.dense as dense
from optimizers.adam import Adam
import layers.relu as relu
import loss
from layers.batch_normalization import BatchNormalization
from utils import training

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
    dense.FCLayer(50),
    relu.ReLu(),
    dense.FCLayer(100),
    relu.ReLu(),
    dense.FCLayer(100),
    relu.ReLu(),
    dense.FCLayer(1),
], Adam, 1e-5)

final_loss = loss.Mse()

epochs = 1000
mini_batch_size = 32
losses = []

#--------------------------------------------------Preprocessing Data-------------------------------------------------------

X_train = training.standardize(X_train, x_train_mean, x_train_std).reshape((X_train.shape + (1,)))
Y_train = training.standardize(Y_train, y_train_mean, y_train_std).reshape((Y_train.shape + (1,)))

X_test = training.standardize(X_test, x_train_mean, x_train_std).reshape((X_test.shape + (1,)))
Y_test = Y_test.reshape(Y_test.shape + (1,1))
#-----------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------Training Model-------------------------------------------------------
for e in range(epochs):
    # shuffle / rebuild mini‑batches each epoch
    mini_batches = training.make_mini_batches(X_train, Y_train, mini_batch_size)
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
prediction = training.destandardize(prediction, y_train_mean, y_train_std)
training_loss = final_loss.root_mse(prediction, Y_train)

prediction_test = network.predict(X_test)
prediction_test = training.destandardize(prediction_test, y_train_mean, y_train_std)
test_loss = final_loss.root_mse(prediction_test, Y_test)

print("training loss", training_loss)
print("test loss", test_loss)
print("avg pred", prediction.mean(), "avg target", Y_train.mean())
print(prediction[:5].flatten())
print(Y_train[:5])

plt.figure()
plt.title("losses over time")
plt.xlabel("epochs")
plt.ylabel("losses")
plt.plot(losses)  # losses

plt.figure()

scatter_data = np.hstack((X_train[:, 6], X_train[:, 7]))
scatter_data = np.hstack((prediction.reshape(prediction.shape[0], 1), scatter_data))
scatter_data = pd.DataFrame(scatter_data, index=None, columns=["preds", "Latitude", "Longitude"])

sns.scatterplot(
    data=scatter_data,
    x="Longitude",
    y="Latitude",
    hue="preds",
    alpha=0.5,
)
plt.legend(title="MedHouseVal", bbox_to_anchor=(1.05, 0.95), loc="upper left")
_ = plt.title("Median house value depending of\n their spatial location")

plt.show()

#-----------------------------------------------------------------------------------------------------------------------
