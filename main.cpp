#include <iostream>

#include "optimizers/cpp_optimizers/headers/adam.h"
#include "layers/cpp_layers/headers/dense.h"

using namespace std;

int main() {

    cout << "helo world";

    /*
    network = Model([
    dense.FCLayer(50),
    relu.ReLu(),
    dense.FCLayer(100),
    relu.ReLu(),
    dense.FCLayer(100),
    relu.ReLu(),
    dense.FCLayer(1),
    ], Adam, 1e-5)
    */

    FCLayer full_connected_layer1(50);
    FCLayer full_connected_layer2(100);
    FCLayer full_connected_layer3(1);

    return 0;
}