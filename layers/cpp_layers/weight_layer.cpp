
#include "headers/weight_layer.h"
#include "cpp_optimizers/headers/base_optimizer.h"

class WeightLayer : public Layer {
public:

    BaseOptimizer *m_optimizer_weights;
    BaseOptimizer *m_optimizer_biases;

    template <typename BaseOptimizer, typename... Args>
    void set_optimizer(BaseOptimizer optimizer, Args... args) {
        m_optimizer_weights = optimizer(args);
        m_optimizer_biases = optimizer(args);

    };

};
