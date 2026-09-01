#include <algorithm>
#include <iostream>
#include <iterator>
#include <Eigen/Dense>
#include <optional>
#include <typeinfo>
#include <random>

#include "headers/training.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;
using namespace std;


MatrixXd standardize(MatrixXd data, std::optional<VectorXd> mean, std::optional<VectorXd> standard_deviation) {
    if (!mean.has_value()) {
        mean = data.colwise().mean();
    }

    int num_rows = static_cast<int>(data.rows());

    if (!standard_deviation.has_value()) {
        standard_deviation = sqrt((data.rowwise() - mean.value().transpose()).array().square().colwise().sum() / num_rows).transpose();
    }
    return ((data.rowwise() - mean.value().transpose()).array().rowwise() / standard_deviation.value().transpose().array()).matrix();
}

MatrixXd destandardize(MatrixXd data, std::optional<VectorXd> mean, std::optional<VectorXd> standard_deviation) {
    if (!mean.has_value()) {
        mean = data.colwise().mean();
    }

    int num_rows = static_cast<int>(data.rows());

    if (!standard_deviation.has_value()) {
        standard_deviation = sqrt((data.rowwise() - mean.value().transpose()).array().square().colwise().sum() / num_rows).transpose();
    }
    MatrixXd temp = (((data.array().rowwise() * standard_deviation.value().transpose().array()).array()).rowwise() + mean.value().transpose().array()).matrix();
    
    return data;
}

//col indexed starting from 1, not 0
MatrixXd remove_column(MatrixXd& matrix, int col) {
    MatrixXd new_matrix(matrix.rows(), matrix.cols() - 1);
    
    if (col < 1 || col > matrix.cols()) {
            throw std::out_of_range("column out of range");
    }

    int left_cols = col - 1;
    int right_cols = matrix.cols() - left_cols - 1;


    if (left_cols > 0) {
        new_matrix.block(0, 0, matrix.rows(), left_cols) = matrix.block(0, 0, matrix.rows(), left_cols);
    }

    if (right_cols > 0) {
        new_matrix.block(0, left_cols + 1, matrix.rows(), right_cols) = matrix.block(0, left_cols + 1, matrix.rows(), right_cols);
    }

    return new_matrix;
}


vector<vector<MatrixXd>> make_mini_batches(MatrixXd x, MatrixXd y, int batch_size) {
    vector<vector<MatrixXd>> mini_batches;

    std::random_device rd;
    std::mt19937 g(rd());

    MatrixXd data(x.rows(), x.cols() + y.cols());
    data << x, y;

    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(x.rows());
    perm.setIdentity();
    std::shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size(), g);
    data = perm * data;
    
    cout << data.rows() / batch_size << endl;
    cout << data.rows() << endl;
    int counter = 0;
    for (int i = 0; i <= data.rows() / batch_size; i++) {
        MatrixXd mini_batch  = data.block(i * batch_size, 0, (data.rows() - i * batch_size), data.cols() - 1);
        if (i != data.rows() / batch_size) {
            mini_batch = data.block(i * batch_size, 0, batch_size, data.cols() - 1);
            counter += batch_size;
        }
        cout << i << endl;
        //cout << mini_batch.cols() << endl;
        //cout << mini_batch.rows() << endl;
        
        MatrixXd x_mini = remove_column(mini_batch, mini_batch.cols());
        MatrixXd y_mini = mini_batch.col(mini_batch.cols() - 1);
        //cout << x_mini << endl;
        //cout << y_mini << endl;
        
        vector<MatrixXd> pair = {x_mini, y_mini};
        mini_batches.push_back(pair);
    }
    cout << "counter" << counter << endl;
    return mini_batches;
}
