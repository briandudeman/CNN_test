#ifndef ADAM
#define ADAM


#include <optional>

#include "base_optimizer.h"

class Adam : public BaseOptimizer {
public:

    Adam(double lr, double beta_one = 0.9, double beta_two = 0.999, double epsilon = 1e-8);

    Eigen::MatrixXd backward(Eigen::MatrixXd grad, int step);

protected:
    std::optional<Eigen::MatrixXd> m_m;
    std::optional<Eigen::MatrixXd> m_v;

    double m_beta_one;
    double m_beta_two;
    double m_epsilon;

};

#endif