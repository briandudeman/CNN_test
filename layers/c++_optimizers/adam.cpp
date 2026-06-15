
#include <iostream>
#include <Eigen/Dense>

#include "headers/base_optimizer.h"

class Adam : BaseOptimizer {
public:

    Adam(double lr, double beta_one = 0.9, double beta_two = 0.999, double epsilon = 1e-8) : BaseOptimizer(lr) {
        m_beta_one = beta_one;
        m_beta_two = beta_two;
        m_epsilon = epsilon;
    };

    

}

    def backward(self, grad, step):
        #print(grad.shape)
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)

        self.m = self.beta_one * self.m + (1 - self.beta_one) * grad
        self.v = self.beta_two * self.v + (1 - self.beta_two) * np.square(grad)

        m_hat = self.m / (1 - self.beta_one ** step)
        v_hat = self.v / (1 - self.beta_two ** step)
        #print(self.lr * (m_hat / (np.sqrt(v_hat) + self.epsilon)))
        return self.lr * (m_hat / (np.sqrt(v_hat) + self.epsilon))

