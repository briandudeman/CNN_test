#include <iostream>
#include <Eigen/Dense>
#include <optional>
#include <typeinfo>

#include "headers/training.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;


MatrixXd standardize(MatrixXd data, std::optional<VectorXd> mean, std::optional<VectorXd> standard_deviation) {
    if (!mean.has_value()) {
        mean = data.colwise().mean();
    }

    int num_rows = static_cast<int>(data.rows());

    if (!standard_deviation.has_value()) {
        standard_deviation = sqrt((data.rowwise() - mean.value().transpose()).array().square().colwise().sum() / num_rows).transpose();
    }
    return ((data.rowwise() - mean.value().transpose()).array().rowwise() / standard_deviation.value().transpose().array()).matrix();
}

MatrixXd destandardize(MatrixXd data, std::optional<VectorXd> mean, std::optional<VectorXd> standard_deviation) {
    if (!mean.has_value()) {
        mean = data.colwise().mean();
    }

    int num_rows = static_cast<int>(data.rows());

    if (!standard_deviation.has_value()) {
        standard_deviation = sqrt((data.rowwise() - mean.value().transpose()).array().square().colwise().sum() / num_rows).transpose();
    }
    MatrixXd temp = (((data.array().rowwise() * standard_deviation.value().transpose().array()).array()) + mean.value().transpose().array()).matrix();
    //std::cout << data.array().rowwise() * standard_deviation.value().transpose().array() << endl;
    std::cout << standard_deviation.value().transpose().rows() << endl;
    std::cout << standard_deviation.value().transpose().cols() << endl;
    
    return data;
}
