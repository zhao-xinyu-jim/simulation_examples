#ifndef SOLVER_H_
#define SOLVER_H_
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <Eigen/SparseQR>
#include <eigen3/unsupported/Eigen/KroneckerProduct>

#include <sophus/so2.hpp>
#include <iostream>
#include <vector>
#include <deque>
#include <map>
struct comparestruct
{
    std::deque<double> a1;
    std::deque<double> a2;
    double lower_bound;
    double upper_bound;
    double cost;
};
struct Lower_Upper_Bound
{
    double lower;
    double upper;
};
class Solver
{
private:
    int feature_size_;
    int pose_size_;

    double c1_ = 0;
    double gamma_ = 1;

    Eigen::MatrixXd b_;
    Eigen::MatrixXd phi_;
    Eigen::MatrixXd beta_;
    Eigen::MatrixXd phi_lowercase_;

    Eigen::MatrixXd C_;
    Eigen::MatrixXd A_;
    std::map<int, Eigen::MatrixXd> Ai_;
    std::map<int, Eigen::MatrixXd> Qi_;

    Eigen::VectorXd z_xy_, z_ori_;
    std::map<int, Eigen::VectorXd> z_xy_i_;
    std::map<int, Eigen::VectorXd> z_xy_i_perp_;

    std::map<int, double> init_phi_;

public:
    Solver(){};
    Solver(int feature_size, int pose_size);
    void release();

    void SetInitialValue(std::map<int, double> &init_phi);
    void SetA(Eigen::MatrixXd &A);
    void SetAi(Eigen::MatrixXd &Ai, int i);
    void SetAi(std::map<uint, Eigen::MatrixXd> &Ai);
    // void SetQi(Eigen::MatrixXd &Qi, int i);
    void SetZ(Eigen::VectorXd &z_xy, Eigen::VectorXd &z_ori);
    void SetZi(Eigen::VectorXd &zi, int i);
    void SetZi(std::map<uint, Eigen::VectorXd> &Zi);
    void SetC(Eigen::MatrixXd &C);
    void SetGama(double gama);
    // void SetZiperp(Eigen::VectorXd &zi_perp, int i);
    void ComputeParam();
    double ComputeCost(std::map<int, double> &phi);
    double ComputeCost(std::vector<double> &phi);
    std::vector<double> solve(int iters);

private:
    void ComputeBound(std::map<int, double> &phi1, std::map<int, double> &phi2, double &lower_bound, double &upper_bound);
    void ComputeBound(comparestruct &phis, double &lower_bound, double &upper_bound);
    void ComputeBound(comparestruct &phis);
    void ComputeBound1(comparestruct &phis, int divide_size);
    void GeneratePhi(double begin, double end, int devide_size, std::vector<std::vector<double>> &data);
    void GeneratePhi(std::deque<double> &begin, std::deque<double> &end, int devide_size, std::vector<std::vector<double>> &data);
    void iter(int layer, std::vector<std::vector<double>> &data, comparestruct &cpst, std::vector<comparestruct> &cpsts);
    void iter1(int layer, std::vector<std::vector<double>> &data, std::vector<double> &temp, std::vector<std::vector<double>> &results);
    Eigen::Matrix2d GetR(double phi);
    Eigen::MatrixXd GetRinvhat(std::vector<double> &phis);
    Eigen::VectorXd ComputeZperp(Eigen::VectorXd &z, int size);
    void MaxMinCos(double phi_min, double phi_max, double &max, double &min);
    int CompareRange(double &max1, double &min1, double &max2, double &min2);

    void Computec1();
    void ComputeQi();
    void ComputePhi();
    void Computeb();
    void Computebeta();
};

#endif