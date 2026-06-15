#ifndef BASE_OPTIMIZER
#define BASE_OPTIMIZER



class BaseOptimizer {
public:

    float m_lr;

    BaseOptimizer(float lr) {
        m_lr = lr;
    };

    virtual void backward() = 0;

};
#endif