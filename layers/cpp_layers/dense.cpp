#include "iostream"
#include "headers/dense.h"
#include "headers/weight_layer.h"

#include <Eigen/Dense>

using namespace std;

FCLayer::FCLayer(int num_nodes) : WeightLayer() {
    m_m = num_nodes;
};

MatrixXd FCLayer::forward(MatrixXd input) {
    m_input = input;
    m_d = input.cols();
    m_n = input.rows();
    
    if (!m_weights.has_value()) {
        m_x_limit = sqrt((2.0 / (m_n + m_m)));
        m_weights = MatrixXd::Random(m_m, m_d) * m_x_limit;
        m_biases = MatrixXd::Random(m_m, 1) * m_x_limit;
    };

    m_z = m_weights.value() * (m_input) + m_biases.value();


    return m_z;

};

MatrixXd FCLayer::backward(MatrixXd dLdZ, int step, float lr) {
    
    m_dLdA = m_weights.value().transpose() * dLdZ;
    
    m_dLdW = dLdZ * m_input.transpose();
    m_dLdW0 = dLdZ.colwise().sum();

    m_weights.value() -= m_optimizer_weights->backward(m_dLdW, step);
    m_biases.value() -= m_optimizer_biases->backward(m_dLdW0, step);
    
    return m_dLdA;
};
