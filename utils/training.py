import numpy as np


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


def predict(network, input):
    output = input
    for i, layer in enumerate(network):
        #print("output shape ", i, " ", output.shape)
        output = layer.forward(output)
    return output


