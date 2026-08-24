#ifndef LAYER
#define LAYER

#include <Eigen/Dense>

using Eigen::MatrixXd;

class Layer{
public:

    float m_lr = 0.0001f;


    virtual MatrixXd forward(MatrixXd input) = 0;
    virtual MatrixXd backward(MatrixXd dLdZ, int step, float lr=.00001) = 0;

};

#endif