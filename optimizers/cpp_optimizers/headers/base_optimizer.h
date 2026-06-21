#ifndef BASE_OPTIMIZER
#define BASE_OPTIMIZER

#include <Eigen/Dense>


class BaseOptimizer {
public:

    float m_lr;

    BaseOptimizer(float lr) {
        m_lr = lr;
    };

    virtual ~BaseOptimizer() {};

    virtual Eigen::MatrixXd backward(Eigen::MatrixXd grad, int step) = 0;

};
#endif