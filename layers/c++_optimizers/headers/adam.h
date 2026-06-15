#ifndef ADAM
#define ADAM

#include "base_optimizer.h"

class Adam : BaseOptimizer {
public:

    double m_beta_one;
    double m_beta_two;
    double m_epsilon;
    double m_m;
    double m_v;

    Adam(double lr, double beta_one = 0.9, double beta_two = 0.999, double epsilon = 1e-8) : BaseOptimizer(lr) {};

};

#endif