from layers.layer import Layer

class WeightLayer(Layer):
    def __init__(self):
        super().__init__()

    def set_optimizer(self, optimizer, *optimizer_args):
        self.optimizer_weights = optimizer(*optimizer_args)
        self.optimizer_biases = optimizer(*optimizer_args)