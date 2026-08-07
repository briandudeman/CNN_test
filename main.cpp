#include <iostream>
#include <sstream>
#include <fstream>
#include <Eigen/Dense>
#include <vector>

#include "optimizers/cpp_optimizers/headers/adam.h"
#include "layers/cpp_layers/headers/dense.h"
#include "layers/cpp_layers/headers/relu.h"
#include "layers/cpp_layers/headers/layer.h"
#include "model.h"

using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using namespace std;

int main() {

    string filename{"housing.csv"};
    ifstream input{filename};

    if (!input.is_open()) {
        std::cerr << "Couldn't read file: " << filename << "\n";
        return 1; 
    }

    vector<vector<double>> dataset_vec;

    string line;
    getline(input, line); // ignoring first line with strings

    for (;getline(input, line);) {
        istringstream ss(move(line));

        vector<double> row;
        if (!dataset_vec.empty()){
            row.reserve(dataset_vec.front().size());
        }

        for (string value; getline(ss, value, ',');) {
            try {
                row.push_back(stod(value));
            } catch (const invalid_argument& e) {
                ;
            }
        }

        dataset_vec.push_back(row);
    }


    int rows = dataset_vec.size();
    int cols = dataset_vec[0].size();

    MatrixXd dataset(rows, cols);

    for (int i = 0; i < rows; i++) {
        dataset.row(i) = RowVectorXd::Map(dataset_vec[i].data(), cols);
    }

    cout << dataset.rows() << endl;
    cout << dataset.cols() << endl;

    vector<std::unique_ptr<Layer>> model_vec;

    model_vec.push_back(std::make_unique<FCLayer>(50));
    model_vec.push_back(std::make_unique<ReLu>());
    model_vec.push_back(std::make_unique<FCLayer>(100));
    model_vec.push_back(std::make_unique<ReLu>());
    model_vec.push_back(std::make_unique<FCLayer>(100));
    model_vec.push_back(std::make_unique<ReLu>());
    model_vec.push_back(std::make_unique<FCLayer>(1));

    Model model = Model::create<Adam>(model_vec, 0.00001);


    return 0;
}