
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

#include "headers/adam.h"

using Eigen::MatrixXd;
using namespace std;

Adam::Adam(double lr, double beta_one, double beta_two, double epsilon) : BaseOptimizer(lr), m_m(), m_v(), m_initialized(false) {
    m_beta_one = beta_one;
    m_beta_two = beta_two;
    m_epsilon = epsilon;
};

MatrixXd Adam::backward(MatrixXd grad, int step) {
    if (!m_initialized) {
        m_m = MatrixXd::Zero(grad.rows(), grad.cols());
        m_v = MatrixXd::Zero(grad.rows(), grad.cols());
        m_initialized = true;
    }


    m_m = m_beta_one * m_m + (1 - m_beta_one) * grad;
    m_v = m_beta_two * m_v + (1 - m_beta_two) * grad.cwiseSqrt();

    MatrixXd m_hat = m_m.array() / MatrixXd::Constant(m_m.rows(), m_m.cols(), (1 - pow(m_beta_one, step))).array();
    MatrixXd v_hat = m_v.array() / MatrixXd::Constant(m_v.rows(), m_v.cols(), (1 - pow(m_beta_two, step))).array();
    return m_lr * (m_hat.array() / (v_hat.cwiseSqrt() + MatrixXd::Constant(v_hat.rows(), v_hat.cols(), m_epsilon)).array());
};

/*
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
*/
