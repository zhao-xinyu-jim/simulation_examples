#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <Eigen/SparseQR>
#include <eigen3/unsupported/Eigen/KroneckerProduct>

#include <sophus/so2.hpp>
#include <yaml-cpp/yaml.h>

#include <iostream>
#include <vector>
#include <deque>
#include <map>
#include <set>

class PoseNode;

struct comparestruct
{
    std::deque<double> a1;
    std::deque<double> a2;
    double lower_bound;
    double upper_bound;
    double cost;
};

class Optimizer
{
public:
    Optimizer(/* args */);
    void AddNodes(std::map<int, std::shared_ptr<PoseNode>> &nodes);
    void ComputeParam();
    void SetInformation(double x, double y, double theta);
    void SetFixedId(int id);
    double ObjFunc(std::map<int, double> &phis);
    int CompareRange(double &max1, double &min1, double &max2, double &min2);
    std::map<int, double> solve();
    
private:
    void ComputeBound(comparestruct &phis);
    void ComputeBound1(comparestruct &phis,int divide_size=4);
    void MaxMinCos(double phi_0, double phi_1, double &max, double &min);
    void iter(int layer, std::vector<std::vector<double>> &data, comparestruct &cpst, std::vector<comparestruct> &cpsts);
    void iter1(int layer, std::vector<std::vector<double>> &data, std::vector<double> &temp, std::vector<std::vector<double>> &results);
    void GeneratePhi(double begin, double end, int devide_size, std::vector<std::vector<double>> &data);
    void GeneratePhi(std::deque<double> &begin, std::deque<double> &end, int devide_size, std::vector<std::vector<double>> &data);
    Eigen::MatrixXd ComputeX(std::map<int, double> &phis);

private:
    /* data */
    int fixed_id_=0;
    double inf_x_, inf_y_, inf_theta_;

    double c1_;
    Eigen::MatrixXd bij_;
    Eigen::MatrixXd betaij0_;
    Eigen::MatrixXd phiij0_;
    Eigen::MatrixXd philowij0_;

    Eigen::SparseMatrix<double> ACA_inv_AC_spa_;
    Eigen::SparseMatrix<double> z_spa_;

    std::map<int, std::shared_ptr<PoseNode>> nodes_;
};

#endif