#include <iostream>
#include <Eigen/Dense>
#include "mse.h"


using Eigen::MatrixXd;
using namespace std;

double Mse::root_mse(MatrixXd y_pred, MatrixXd y) {
    MatrixXd difference_squared = (y.array() - y_pred.array()).pow(2);
    return sqrt(difference_squared.sum() / y_pred.rows());    
}

MatrixXd Mse::mse(MatrixXd y_pred, MatrixXd y) {
    return y;
}

MatrixXd Mse::mse_prime(MatrixXd y_pred, MatrixXd y) {
    return y;
}