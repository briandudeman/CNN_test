# Testing Fully Connected Layers

To test my implementation of a regular fully connected neural network layer, as well as the ADAM optimizer, ReLu function, and MSE loss, I used sci-kit learn's [California housing datset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) for regression. This dataset comprises of 20640 houses, each with 8 attributes, all numeric:
- MedInc: median income in block group
- HouseAge: median house age in block group
- AveRooms: average number of rooms per household
- AveBedrms: average number of bedrooms per household
- Population: block group population
- AveOccup: average number of household members
- Latitude: block group latitude
- Longitude: block group longitude

This is all used to predict the price of each house, in hundreds of thousands of dollars.

## Model
I used a model with 4 layers in total, with 50, 100, 100, and 1 neuron(s), respectively. The first 3 of these then had ReLu activation layers after them:
```
network = Model([
    dense.FCLayer(50),
    relu.ReLu(),
    dense.FCLayer(100),
    relu.ReLu(),
    dense.FCLayer(100),
    relu.ReLu(),
    dense.FCLayer(1),
], Adam, 1e-5)
```
As you can also see, I also used ADAM as an optimizer, as well as a starting learning rate of 1e-5.

## Results and Comparision
In addition to evaluating my model based on loss, I also compared it to an implementation using TensorFlow:

```
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(50, activation='relu'),
  tf.keras.layers.Dense(100, activation='relu'),
  tf.keras.layers.Dense(100, activation='relu'),
  tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse') 
```

Both of these models were trained for 1000 epochs, on standardized data with a 0.2 testing split from the original dataset. The results of each model can be seen below:

<div style="margin-left: auto; margin-right: auto; width: 80%">

| Model         | Training Time (s) | Training Loss (MSE) | Test Loss (MSE) |
|--------------|-----|-----------|----|
| TensorFlow |  ~654.967 |        0.0282 | 0.392 |
| Mine v.1      |  ~2295.930 |          0.481 | .529 |
</div>

