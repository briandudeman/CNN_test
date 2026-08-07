#ifndef MODEL

#define MODEL

#include <iostream>
#include <vector>
#include <memory>
#include <utility>
#include <Eigen/Dense>

#include "layers/cpp_layers/headers/layer.h"
#include "layers/cpp_layers/headers/weight_layer.h"
#include "optimizers/cpp_optimizers/headers/base_optimizer.h"

using Eigen::MatrixXd;
using namespace std;

template<typename Base, typename T>
inline bool instanceof(const T *ptr) {
   return dynamic_cast<const Base*>(ptr) != nullptr;
}

class Model {

public:

    Model() = default;

    // Model owns its layers via unique_ptr, so it cannot be copied,
    // only moved.
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) = default;
    Model& operator=(Model&&) = default;

    template <typename OptimizerType, typename... OptimizerArgs>
    static Model create(vector<unique_ptr<Layer>>& layers, OptimizerArgs&&... optimizer_args) {
        Model m;
        m.m_layers = std::move(layers);

        for (auto& l : m.m_layers) {
            if (WeightLayer* w_l = dynamic_cast<WeightLayer*>(l.get())) {
                w_l->set_optimizer<OptimizerType>(std::forward<OptimizerArgs>(optimizer_args)...);
            }
        }

        return m;
    };

    MatrixXd predict(MatrixXd input);

    void backprop(MatrixXd grad, int epoch);

protected:

    vector<unique_ptr<Layer>> m_layers;

};

#endif