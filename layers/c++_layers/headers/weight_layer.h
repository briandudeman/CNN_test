#ifndef WEIGHTLAYER
#define WEIGHTLAYER

#include "layer.h"

class WeightLayer : public Layer{
public:

    WeightLayer();

};


    def __init__(self):
        super().__init__()

    def set_optimizer(self, optimizer, *optimizer_args):
        self.optimizer_weights = optimizer(*optimizer_args)
        self.optimizer_biases = optimizer(*optimizer_args)

#endif