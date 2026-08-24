#ifndef TRAINING
#define TRAINING

#include <Eigen/Dense>
#include <iostream>
#include <optional>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

MatrixXd standardize(MatrixXd data, std::optional<VectorXd> mean = std::nullopt, std::optional<VectorXd> standard_deviation = std::nullopt);


MatrixXd destandardize(MatrixXd data, std::optional<VectorXd> mean = std::nullopt, std::optional<VectorXd> standard_deviation = std::nullopt);

#endif