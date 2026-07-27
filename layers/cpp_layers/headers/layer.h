#ifndef LAYER
#define LAYER

#include <Eigen/Dense>

using Eigen::MatrixXd;

class Layer{
public:



    virtual MatrixXd forward(MatrixXd input) = 0;
    virtual MatrixXd backward(MatrixXd dLdZ, int step, float lr=.00001) = 0;

protected:
    int lr = 0.0001;
};

#endif