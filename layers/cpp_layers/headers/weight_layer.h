#ifndef WEIGHTLAYER
#define WEIGHTLAYER

#include "layer.h"
#include "../../../optimizers/cpp_optimizers/headers/base_optimizer.h"


class WeightLayer : public Layer {
public:

    std::unique_ptr<BaseOptimizer> m_optimizer_weights;
    std::unique_ptr<BaseOptimizer> m_optimizer_biases;


    template <typename OptimizerType, typename... Args>
    void set_optimizer(Args... args) {
        m_optimizer_weights = std::make_unique<OptimizerType>(args...);
        m_optimizer_biases = std::make_unique<OptimizerType>(args...);
    };
};


#endif