#include <iostream>
#include <Eigen/Dense>

#include "NelderMeadOptimizer.h"

double function(Eigen::Matrix<double, 3, 1> x){
    Vector<3> target(2,1,3);
    Vector<3> dist = target-x;
    return dist.dot(dist);
}

int main() {

    Eigen::Matrix<double, 3, 1> start(0.0,0.0,0.0);

    auto res = Nelder_Mead_Optimizer<3>(function, start, 0.1, 10e-10);
    std::cout <<"result:  "<<res<<std::endl;
    return 0;
}
