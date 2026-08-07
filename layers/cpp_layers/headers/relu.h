#ifndef RELU
#define RELU

#include <optional>
#include <Eigen/Dense>
#include "layer.h"

using Eigen::MatrixXd;

class ReLu : public Layer {
public:

    MatrixXd forward(MatrixXd z);

    MatrixXd backward(MatrixXd dLdZ, int step, float lr=.00001);


protected:
    std::optional<MatrixXd> m_z;
    std::optional<MatrixXd> m_dLdZ;
};



#endif



