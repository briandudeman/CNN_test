#ifndef FCLAYER
#define FCLAYER

#include <optional>
#include <Eigen/Dense>


#include "weight_layer.h"

using Eigen::MatrixXd;

class FCLayer : public WeightLayer {
public:


    FCLayer(int num_nodes);

    MatrixXd forward(MatrixXd input);

    MatrixXd backward(MatrixXd dLdZ, int step, float lr=.00001);

protected:

    int m_m;
    std::optional<MatrixXd> m_weights;
    std::optional<MatrixXd> m_biases;

    MatrixXd m_input;
    int m_d;
    int m_n;
    float m_x_limit;

    MatrixXd m_z;

    MatrixXd m_dLdZ;
    MatrixXd m_dLdA;
    MatrixXd m_dLdW;
    MatrixXd m_dLdW0;


};
#endif