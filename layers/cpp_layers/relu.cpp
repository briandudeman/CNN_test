
#include <Eigen/Dense>

#include "headers/relu.h"

using Eigen::MatrixXd;

MatrixXd ReLu::forward(MatrixXd z) {
    m_z = z;

    Eigen::ArrayXXd z_array = (z.array()).cast<double>();

    // matrices of 1s and 0s for if greater than 0 or not
    Eigen::ArrayXXd selection = (z.array() >= 0.0).cast<double>();
    Eigen::ArrayXXd inverse_selection = (z.array() < 0.0).cast<double>() * 0.1;

    z_array *= (selection + inverse_selection);

    return z_array.matrix();
};

MatrixXd ReLu::backward(MatrixXd dLdZ, int step, float lr) {
    
    m_dLdZ = dLdZ;
    
    Eigen::ArrayXXd dLdZ_array = (dLdZ.array()).cast<double>();

    // matrices of 1s and 0s for if greater than 0 or not
    Eigen::ArrayXXd selection = (m_z.value().array() >= 0.0).cast<double>();
    Eigen::ArrayXXd inverse_selection = (m_z.value().array() < 0.0).cast<double>() * 0.1;

    dLdZ_array *= (selection + inverse_selection);

    return dLdZ_array.matrix();
};


