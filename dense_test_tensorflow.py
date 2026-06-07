from model import Model
import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf
import keras
import time

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


model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(50, activation='relu'),
  tf.keras.layers.Dense(100, activation='relu'),
  tf.keras.layers.Dense(100, activation='relu'),
  tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse') 


epochs = 1000
mini_batch_size = 32
losses = []

#--------------------------------------------------Preprocessing Data-------------------------------------------------------

X_train = training.standardize(X_train, x_train_mean, x_train_std)
Y_train = training.standardize(Y_train, y_train_mean, y_train_std)

X_test = training.standardize(X_test, x_train_mean, x_train_std)
#-----------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------Training Model-------------------------------------------------------

training_start = time.time()

losses = model.fit(X_train, Y_train,
                   
                   # it will use 'batch_size' number
                   # of examples per example
                   batch_size=mini_batch_size, 
                   epochs=epochs,  # total epoch

                   )

training_end = time.time()

training_time = training_end - training_start
#-----------------------------------------------------------------------------------------------------------------------

#----------------------------------------------Plotting Data----------------------------------------------------------
# evaluate on original scale
prediction = model.predict(X_train)
prediction = training.destandardize(prediction, y_train_mean, y_train_std)
Y_train = training.destandardize(Y_train, y_train_mean, y_train_std)
training_loss = keras.metrics.mean_squared_error(Y_train.reshape((Y_train.shape[0], )), prediction.reshape((prediction.shape[0], )))

print(Y_train.shape)
print(training_loss)
prediction_test = model.predict(X_test)
prediction_test = training.destandardize(prediction_test, y_train_mean, y_train_std)
test_loss = keras.metrics.mean_squared_error(Y_test, prediction_test.reshape((prediction_test.shape[0], )))
print(test_loss)

print("training time, in seconds:", training_time)
print("training loss", training_loss.numpy())
print("test loss", test_loss.numpy())
print("avg prediction", prediction_test.mean(), "avg target", Y_test.mean(), "\n")
print("prediction head", prediction_test[:5].flatten())
print("target head", Y_test[:5].flatten())

plt.figure()
plt.title("losses over time")
plt.xlabel("epochs")
plt.ylabel("losses")
plt.plot(losses.history['loss'], label='Training Loss') 
plt.legend()
plt.figure()

scatter_data = np.hstack((X_test[:, 6].T.reshape((X_test[:, 6].T.shape + (1, ))), X_test[:, 7].T.reshape((X_test[:, 7].T.shape + (1, )))))

scatter_data = np.hstack((prediction_test, scatter_data))
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
