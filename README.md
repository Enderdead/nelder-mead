# nelder-mead

Cpp implementation of the Nelder-Mead optimization algorithm.
Inspired by : https://github.com/fchollet/nelder-mead.
That's make you able to try it very quickly on your cpp project which use Eigen as matrix computing library.


## Example.cpp
First, you have to define your loss function that take an Eigen::VectorX as input. You can extract all feature you want from it with direct access. Here it's an example with a distance calculation.
```cpp
double function(Eigen::Matrix<double, 3, 1> x){
    Vector<3> target(2,1,3);
    Vector<3> dist = target-x;
    return dist.dot(dist);
}
```
After, define your start point according to your input dimenssion.
```cpp
Eigen::Matrix<double, 3, 1> start(0.0,0.0,0.0);
``` 
Then just call the Nelder-Mead function with the right template argument corresponding to your input dimension. You can check at Wikipedia page to understand all constants influances.
```cpp
auto res = Nelder_Mead_Optimizer<3>(function, start, 0.1, 10e-10);
```

## Reference 

See the description of the Nelder-Mead algorithm on Wikipedia: https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
