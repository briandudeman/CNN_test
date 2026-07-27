#ifndef RELU
#define RELU

#include <Eigen/Dense>
#include "layer.h"

using Eigen::MatrixXd

class ReLu : Layer {
public:

    MatrixXd forward(MatrixXd z);

    MatrixXd backward(MatrixXd dLdA, int step);


protected:
    std::optional<MatrixXd> m_z;
    std::optional<MatrixXd> m_dLdA;
}



#endif



