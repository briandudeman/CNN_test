#ifndef MSE
#define MSE


#include <Eigen/Dense>

using Eigen::MatrixXd;

class Mse {
public:

static double root_mse(MatrixXd y_pred, MatrixXd y);

static MatrixXd mse(MatrixXd y_pred, MatrixXd y);

static MatrixXd mse_prime(MatrixXd y_pred, MatrixXd y);
};

#endif