# simulation_examples
simulation_examples for "A Branch-and-Bound Based Method for Globally Optimal Solution of 2D SLAM Problems"

# 1. Prerequisites
We have conducted our experiments and testings in Ubuntu 18.04

## Eigen3
Eigen3, Follow [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page).

## Sophus
Sophus, Follow [Sophus](https://github.com/strasdat/Sophus)

## yaml-cpp
yaml-cpp, Follow [yaml-cpp](https://github.com/jbeder/yaml-cpp)

## Boost
`sudo apt-get install libboost-dev`

## OpenMP

# 2. Build
```
git clone https://github.com/zhao-xinyu-jim/simulation_examples.git

cd simulation_examples/Code_For_Example1/
mkdir build && cd build
cmake ..
make
cd ..
mkdir -p generated_data/gt && mkdir submap/

cd ../Code_For_Example2/
mkdir build && cd build
cmake ..
make
cd ..
mkdir -p data/f_bound/
mkdir -p data/temp
```

# 3. Run
For Example1 in our paper
```
cd Code_For_Example1/build/
./main
```
You can modify the test content by editing 'Code_For_Example1/config.yaml' and 'Code_For_Example1/Example.txt'

For Example2 in our paper
```
cd Code_For_Example2/build/
./main
```
You can modify the test content by editing 'Code_For_Example2/config.yaml'

For Example3 in our paper
```
cd Code_For_Example1/build/
./main_submap
```
You can modify the test content by editing 'Code_For_Example1/config.yaml'