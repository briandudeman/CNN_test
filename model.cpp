
#include "model.h"
#include <Eigen/Dense>


#include "layers/cpp_layers/headers/layer.h"

using namespace std;
using Eigen::MatrixXd;


MatrixXd Model::predict(MatrixXd input) {
    MatrixXd output = input;

    for (auto& l: m_layers) {
        output = l->forward(output);
    };

    return output;
};


void Model::backprop(MatrixXd grad, int epoch) {

    for (auto l = m_layers.rbegin(); l != m_layers.rend(); ++l) {
        grad = (*l)->backward(grad, epoch, (*l)->m_lr);
    };
};

