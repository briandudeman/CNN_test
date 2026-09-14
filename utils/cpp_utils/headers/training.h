#ifndef TRAINING
#define TRAINING

#include <Eigen/Dense>
#include <iostream>
#include <optional>
#include "../../../layers/cpp_layers/headers/layer.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;
using namespace std;

MatrixXd standardize(MatrixXd data, std::optional<VectorXd> mean = std::nullopt, std::optional<VectorXd> standard_deviation = std::nullopt);

MatrixXd destandardize(MatrixXd data, std::optional<VectorXd> mean = std::nullopt, std::optional<VectorXd> standard_deviation = std::nullopt);

vector<vector<MatrixXd>> make_mini_batches(MatrixXd x, MatrixXd y, int batch_size);

MatrixXd remove_column(MatrixXd& matrix, int col);

MatrixXd predict(vector<std::unique_ptr<Layer>> model, MatrixXd input);

#endif