#include <malloc.h>
#include <cfloat>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include "solver.hpp"
#include <yaml-cpp/yaml.h>
#include <omp.h>
#include "tic_toc.hpp"
#define pi 3.14159265358979323846

bool compare1(Lower_Upper_Bound &a, Lower_Upper_Bound &b)
{
    return a.upper < b.upper;
}
bool compare2(comparestruct &a, comparestruct &b)
{
    return a.upper_bound < b.upper_bound;
}
bool compare3(comparestruct &a, comparestruct &b)
{
    return a.lower_bound < b.lower_bound;
}
bool compare4(comparestruct &a, comparestruct &b)
{
    return a.cost < b.cost;
}

Solver::Solver(int feature_size, int pose_size)
    : feature_size_(feature_size), pose_size_(pose_size)
{
}

void Solver::SetInitialValue(std::map<int, double> &init_phi)
{
    init_phi_ = init_phi;
}

void Solver::SetC(Eigen::MatrixXd &C)
{
    C_ = C;
}

void Solver::SetA(Eigen::MatrixXd &A)
{
    A_ = A;
}

void Solver::SetAi(Eigen::MatrixXd &Ai, int i)
{
    Ai_[i] = Ai;
}
void Solver::SetAi(std::map<uint, Eigen::MatrixXd> &Ai)
{
    for (auto &ai : Ai)
    {
        Ai_[(int)(ai.first)] = ai.second;
    }
}

void Solver::SetZ(Eigen::VectorXd &z_xy, Eigen::VectorXd &z_ori)
{
    z_xy_ = z_xy;
    z_ori_ = z_ori;
}

void Solver::SetZi(Eigen::VectorXd &zi, int i)
{
    z_xy_i_[i] = zi;
    z_xy_i_perp_[i] = ComputeZperp(zi, zi.rows() / 2);
}
void Solver::SetZi(std::map<uint, Eigen::VectorXd> &Zi)
{
    for (auto &zi : Zi)
    {
        z_xy_i_[(int)(zi.first)] = zi.second;
        z_xy_i_perp_[(int)(zi.first)] = ComputeZperp(zi.second, zi.second.rows() / 2);
    }
}

void Solver::SetGama(double gama)
{
    gamma_ = gama;
}

void Solver::ComputeParam()
{
    ComputeQi();
    Computeb();
    ComputePhi();
    Computec1();
    Computebeta();
}

double Solver::ComputeCost(std::map<int, double> &phis)
{
    double f = 0;
    f += c1_;
    for (int i = 0; i < pose_size_ - 1; i++)
        for (int j = i + 1; j < pose_size_; j++)
        {
            double phi = phis[j] - phis[i] - phi_lowercase_(i, j);
            f -= 2 * beta_(i, j) * cos(phi);
        }
    return f;
}
double Solver::ComputeCost(std::vector<double> &phis)
{
    double f = 0;
    f += c1_;
    for (int i = 0; i < pose_size_ - 1; i++)
        for (int j = i + 1; j < pose_size_; j++)
        {
            double phi = phis[j] - phis[i] - phi_lowercase_(i, j);
            f -= 2 * beta_(i, j) * cos(phi);
        }
    return f;
}

void Solver::ComputeBound(std::map<int, double> &phi1, std::map<int, double> &phi2,
                          double &lower_bound, double &upper_bound)
{
    lower_bound = c1_;
    upper_bound = c1_;
    for (int i = 0; i < pose_size_ - 1; i++)
        for (int j = i + 1; j < pose_size_; j++)
        {
            double max, min;
            double phi_1 = phi1[j] - phi1[i] - phi_lowercase_(i, j);
            double phi_2 = phi2[j] - phi2[i] - phi_lowercase_(i, j);
            MaxMinCos(phi_1, phi_2, max, min);
            upper_bound -= 2 * beta_(i, j) * min;
            lower_bound -= 2 * beta_(i, j) * max;
        }
}

void Solver::ComputeBound(comparestruct &phis,
                          double &lower_bound, double &upper_bound)
{
    lower_bound = c1_;
    upper_bound = c1_;
    std::deque<double> &phi1 = phis.a1;
    std::deque<double> &phi2 = phis.a2;
    for (int i = 0; i < pose_size_ - 1; i++)
        for (int j = i + 1; j < pose_size_; j++)
        {
            double max, min;
            double phi_1 = phi1[j] - phi1[i] - phi_lowercase_(i, j);
            double phi_2 = phi2[j] - phi2[i] - phi_lowercase_(i, j);
            MaxMinCos(phi_1, phi_2, max, min);
            upper_bound -= 2 * beta_(i, j) * min;
            lower_bound -= 2 * beta_(i, j) * max;
        }
}

void Solver::ComputeBound(comparestruct &phis)
{
    phis.lower_bound = c1_;
    phis.upper_bound = c1_;
    std::deque<double> &phi1 = phis.a1;
    std::deque<double> &phi2 = phis.a2;
    for (int i = 0; i < pose_size_ - 1; i++)
        for (int j = i + 1; j < pose_size_; j++)
        {
            double max, min;
            double phi_1 = phi1[j] - phi2[i] - phi_lowercase_(i, j);
            double phi_2 = phi2[j] - phi1[i] - phi_lowercase_(i, j);
            MaxMinCos(phi_1, phi_2, max, min);
            phis.upper_bound -= 2 * beta_(i, j) * min;
            phis.lower_bound -= 2 * beta_(i, j) * max;
        }
}
void Solver::ComputeBound1(comparestruct &phis, int divide_size)
{
    std::deque<double> &phi1 = phis.a1;
    std::deque<double> &phi2 = phis.a2;
    std::vector<double> temp;
    std::vector<std::vector<double>> data, results;
    GeneratePhi(phi1, phi2, divide_size, data);
    iter1(0, data, temp, results);
    double max = DBL_MIN, min = DBL_MAX, mid;
    std::vector<double> values;
    for (int t = 0; t < results.size(); t++)
    {
        auto &result = results[t];
        double current = c1_;

        for (int i = 0; i < pose_size_ - 1; i++)
        {
            for (int j = i + 1; j < pose_size_; j++)
            {
                if (i == j)
                    continue;
                double c = cos(result[j] - result[i] - phi_lowercase_(i, j));

                current -= 2 * beta_(i, j) * c;
            }
        }
        values.push_back(current);
    }
    sort(values.begin(), values.end());
    max = values.back();
    min = values[0];
    mid = values[values.size() / 2];
    phis.upper_bound = max + (max - mid) / 2;
    phis.lower_bound = min - (mid - min) / 2;
}

Eigen::Matrix2d Solver::GetR(double phi)
{
    Sophus::SO2d so2 = Sophus::SO2d::exp(phi);
    Eigen::Matrix2d R = so2.matrix();
    return R;
}

Eigen::MatrixXd Solver::GetRinvhat(std::vector<double> &phis)
{
    Eigen::MatrixXd R_invhat = Eigen::MatrixXd::Zero(phis.size() * 2, phis.size() * 2);
    for (int i = 0; i < (int)phis.size(); i++)
        R_invhat.block(i * 2, i * 2, 2, 2) = GetR(phis[i]);
    return R_invhat;
}

Eigen::VectorXd Solver::ComputeZperp(Eigen::VectorXd &z, int size)
{
    std::vector<double> phis(size);
    for (int i = 0; i < size; i++)
        phis[i] = pi / 2;
    Eigen::MatrixXd R_invhat = GetRinvhat(phis);

    Eigen::VectorXd z_perp = R_invhat * z;
    return z_perp;
}

void Solver::MaxMinCos(double phi_0, double phi_1, double &max, double &min)
{
    if (phi_1 - phi_0 >= 2 * pi)
    {
        max = 1;
        min = -1;
        return;
    }
    double cos_phi0 = cos(phi_0);
    double cos_phi1 = cos(phi_1);
    double phi0, phi1;
    if (phi_0 < 0)
        phi0 = phi_0 - pi;
    else
        phi0 = phi_0;

    if (phi_1 < 0)
        phi1 = phi_1 - pi;
    else
        phi1 = phi_1;

    int phase0 = phi0 / pi;
    int phase1 = phi1 / pi;
    if (phase0 == phase1)
    {
        max = cos_phi0 > cos_phi1 ? cos_phi0 : cos_phi1;
        min = cos_phi0 < cos_phi1 ? cos_phi0 : cos_phi1;
        return;
    }
    else if (phase1 - phase0 >= 2)
    {
        max = 1;
        min = -1;
        return;
    }
    else if (phase1 - phase0 == 1)
    {
        if (abs(phase0 % 2) == 1)
        {
            max = 1;
            min = cos_phi0 < cos_phi1 ? cos_phi0 : cos_phi1;
            return;
        }
        else if (abs(phase0 % 2) == 0)
        {
            min = -1;
            max = cos_phi0 > cos_phi1 ? cos_phi0 : cos_phi1;
            return;
        }
    }
}

int Solver::CompareRange(double &max1, double &min1, double &max2, double &min2)
{
    if (max1 <= min2)
    {
        if (max1 < 0 && min2 < 0)
            return 2;
        else if (max1 < 0 && min2 > 0)
            return 0;
        return 1;
    }
    else if (max2 <= min1)
    {
        if (max2 < 0 && min1 < 0)
            return 1;
        else if (max2 < 0 && min1 > 0)
            return 0;
        return 2;
    }
    else
        return 0;
}

void Solver::GeneratePhi(double begin, double end, int devide_size, std::vector<std::vector<double>> &data)
{
    data.resize(pose_size_);
    data[0] = std::vector<double>(devide_size + 1, 0);
    for (int i = 1; i < pose_size_; i++)
    {
        data[i].resize(devide_size + 1);
        for (int j = 0; j <= devide_size; j++)
        {
            data[i][j] = (end - begin) * ((double)j / (double)devide_size) + begin;
        }
    }
}

void Solver::GeneratePhi(std::deque<double> &begin, std::deque<double> &end, int devide_size, std::vector<std::vector<double>> &data)
{
    data.clear();
    data.resize(pose_size_);
    data[0] = std::vector<double>(devide_size + 1, 0);
    for (int i = 1; i < pose_size_; i++)
    {
        data[i].clear();
        data[i].resize(devide_size + 1);
        for (int j = 0; j <= devide_size; j++)
        {
            data[i][j] = (end[i] - begin[i]) * ((double)j / (double)devide_size) + begin[i];
        }
    }
}

void Solver::iter1(int layer, std::vector<std::vector<double>> &data, std::vector<double> &temp, std::vector<std::vector<double>> &results)
{
    int max_layer = data.size() - 1;
    for (size_t i = 0; i < data[layer].size(); i++)
    {
        if (layer == max_layer)
        {
            temp.push_back(data[layer][i]);
            results.push_back(temp);
            temp.pop_back();
        }
        else
        {
            temp.push_back(data[layer][i]);
            iter1(layer + 1, data, temp, results);
            temp.pop_back();
        }
    }
}

void Solver::iter(int layer, std::vector<std::vector<double>> &data, comparestruct &cpst, std::vector<comparestruct> &cpsts)
{
    int max_layer = data.size() - 1;
    for (size_t i = 0; i < data[layer].size() - 1; i++)
    {
        if (layer == max_layer)
        {
            cpst.a1.emplace_back(data[layer][i]);
            cpst.a2.emplace_back(data[layer][i + 1]);
            cpsts.emplace_back(cpst);

            cpst.a1.pop_back();
            cpst.a2.pop_back();
        }
        else
        {
            cpst.a1.emplace_back(data[layer][i]);
            cpst.a2.emplace_back(data[layer][i + 1]);
            iter(layer + 1, data, cpst, cpsts);
            cpst.a1.pop_back();
            cpst.a2.pop_back();
        }
    }
}

void Solver::Computec1()
{
    c1_ = 0;
    c1_ += z_xy_.transpose() * C_ * z_xy_;
    for (int i = 0; i < pose_size_; i++)
        c1_ -= b_(i, i);

    for (int i = 1; i < pose_size_; i++)
        c1_ += 2 * gamma_;
}

void Solver::ComputeQi()
{
    for (auto &Ai : Ai_)
    {
        int i = Ai.first;
        Eigen::MatrixXd &ai = Ai.second;

        Eigen::SparseMatrix<double> a_ = A_.sparseView();
        Eigen::SparseMatrix<double> ai_ = ai.sparseView();
        Eigen::SparseMatrix<double> c_ = C_.sparseView();

        Eigen::MatrixXd ACAT = a_ * c_ * a_.transpose();
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(ACAT.rows(), ACAT.cols());
        Eigen::SparseMatrix<double> acat = ACAT.sparseView();
        Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::AMDOrdering<int>> QRsolver;
        QRsolver.compute(acat);
        Eigen::SparseMatrix<double> ACAT_inv = QRsolver.solve(I.sparseView());

        Eigen::MatrixXd qi = c_ * ai_.transpose() * ACAT_inv * a_ * c_;
        Qi_[i] = qi;
    }
}

void Solver::ComputePhi()
{
    phi_ = Eigen::MatrixXd::Zero(pose_size_, pose_size_);
    std::vector<int> pose_index;
    for (auto &z_xy : z_xy_i_)
        pose_index.emplace_back(z_xy.first);
    for (size_t i = 0; i < pose_index.size() - 1; i++)
        for (size_t j = i + 1; j < pose_index.size(); j++)
            phi_(i, j) = atan2(z_xy_.transpose() * Qi_[pose_index[i]] * z_xy_i_perp_[pose_index[j]], z_xy_.transpose() * Qi_[pose_index[i]] * z_xy_i_[pose_index[j]]);
}

void Solver::Computeb()
{
    b_ = Eigen::MatrixXd::Zero(pose_size_, pose_size_);
    std::vector<int> pose_index;
    for (auto &z_xy : z_xy_i_)
        pose_index.emplace_back(z_xy.first);

    for (size_t i = 0; i < pose_index.size(); i++)
        for (size_t j = i; j < pose_index.size(); j++)
        {
            b_(i, j) = sqrt(pow(z_xy_.transpose() * Qi_[pose_index[i]] * z_xy_i_[pose_index[j]], 2) + pow(z_xy_.transpose() * Qi_[pose_index[i]] * z_xy_i_perp_[pose_index[j]], 2));
        }
}

void Solver::Computebeta()
{
    beta_ = Eigen::MatrixXd::Zero(pose_size_, pose_size_);
    phi_lowercase_ = Eigen::MatrixXd::Zero(pose_size_, pose_size_);
    for (int i = 0; i < pose_size_ - 1; i++)
        for (int j = i + 1; j < pose_size_; j++)
            if (i + 1 == j)
            {
                double bij_2 = b_(i, j) * b_(i, j);
                double gamma_2 = gamma_ * gamma_;
                double complex = 2 * b_(i, j) * gamma_ * cos(phi_(i, j) - z_ori_(j - 1));
                beta_(i, j) = sqrt(bij_2 + gamma_2 + complex);
                phi_lowercase_(i, j) = atan2(b_(i, j) * sin(phi_(i, j)) + gamma_ * sin(z_ori_(j - 1)), b_(i, j) * cos(phi_(i, j)) + gamma_ * cos(z_ori_(j - 1)));
            }
            else
            {
                beta_(i, j) = b_(i, j);
                phi_lowercase_(i, j) = phi_(i, j);
            }
}

void Solver::release()
{
    b_ = Eigen::MatrixXd();
    phi_ = Eigen::MatrixXd();

    C_ = Eigen::MatrixXd();
    A_ = Eigen::MatrixXd();
    std::map<int, Eigen::MatrixXd>().swap(Ai_);
    std::map<int, Eigen::MatrixXd>().swap(Qi_);

    z_xy_ = Eigen::VectorXd();
    z_ori_ = Eigen::VectorXd();
    std::map<int, Eigen::VectorXd>().swap(z_xy_i_);
    std::map<int, Eigen::VectorXd>().swap(z_xy_i_perp_);

    std::map<int, double>().swap(init_phi_);
    malloc_trim(0);
}

std::vector<double> Solver::solve(int iters)
{
    TicToc tic_toc;
    YAML::Node config_node = YAML::LoadFile("../config.yaml");
    int start_random = config_node["start_frame_for_random"].as<int>();
    int threads_num = config_node["threads"].as<int>();
    int divide_size = config_node["divide_size"].as<int>();
    std::string save_time_path = config_node["path"].as<std::string>();
    std::string save_cost_path = config_node["cost_path"].as<std::string>();

    std::ofstream file_time, file_cost;
    std::vector<std::vector<double>> data, rel_data;
    comparestruct cpst;
    std::vector<comparestruct> cpsts, cpsts1, temp_cpsts;
    std::vector<Lower_Upper_Bound> lubs;
    GeneratePhi(-pi, pi, 2, data);
    iter(1, data, cpst, cpsts);
    rel_data.swap(data);

    for (int k = 0; k < iters; k++)
    {
        std::cout << "==========iter:" << k << "==========" << std::endl;
        tic_toc.tic();
        if (k < iters - 1)
        {
            int countss = 0;
            omp_set_num_threads(threads_num);
#pragma omp parallel
            {
#pragma omp for
                for (int i = 0; i < cpsts.size(); i++)
                {
                    auto &cp = cpsts[i];
                    cp.a1.emplace_front(0);
                    cp.a2.emplace_front(0);
                    if (k < start_random)
                        ComputeBound(cp);
                    else
                        ComputeBound1(cp, divide_size);
                    if (k >= start_random)
                    {
                        printf("\r %f %%", 100.0 * (float)(countss++) / (float)(cpsts.size()));
                    }
                }
            }
            if (k >= start_random)
                std::cout << std::endl;
        }
        else
        {
            for (auto &cp : cpsts)
            {
                std::vector<double> phis;
                phis.push_back(0);
                for (size_t i = 0; i < cp.a1.size(); i++)
                {
                    double aa = (cp.a1[i] + cp.a2[i]) / 2.;
                    phis.emplace_back(aa);
                }
                cp.cost = ComputeCost(phis);
            }
            comparestruct &cp = *min_element(cpsts.begin(), cpsts.end(), compare4);
            std::cout << "final_cost: " << cp.cost << std::endl;
            std::vector<double> phis;
            phis.push_back(0);
            for (size_t i = 0; i < cp.a1.size(); i++)
            {
                double aa = (cp.a1[i] + cp.a2[i]) / 2.;
                phis.emplace_back(aa);
            }
            file_time.open(save_time_path.c_str(), std::ios::app);
            file_time << std::endl;
            file_time.close();
            file_cost.open(save_cost_path.c_str(), std::ios::app);
            file_cost << cp.cost << std::endl;
            file_cost.close();
            return phis;
        }

        double timecost = tic_toc.toc();
        std::cout << "time_cost: " << timecost << std::endl;
        file_time.open(save_time_path.c_str(), std::ios::app);
        file_time << timecost << " ";
        file_time.close();

        int count_add = 0;
        int count_earse = 0;
        comparestruct &min_cp = *min_element(cpsts.begin(), cpsts.end(), compare2);
        double min_max = min_cp.upper_bound;
        double min_min = min_cp.lower_bound;

        std::vector<comparestruct>().swap(temp_cpsts);
        std::cout << "min_max:" << min_max << std::endl;
        std::cout << "min_min:" << min_min << std::endl;

        for (auto iter = cpsts.begin(); iter != cpsts.end(); iter++)
        {
            if (iter->lower_bound > min_max) //
                count_earse++;
            else
            {
                cpsts1.emplace_back(*iter);
                count_add++;
            }
        }
        std::vector<comparestruct>().swap(cpsts);

        std::cout << "count_add:" << count_add << std::endl;
        std::cout << "count_earse:" << count_earse << std::endl;

        temp_cpsts.clear();
        for (auto &cp : cpsts1)
        {
            std::vector<comparestruct> temp_cpst, rel_cpst;
            std::vector<std::vector<double>> t_data, rel_t_data;

            if (k < iters)
            {
                GeneratePhi(cp.a1, cp.a2, 2, t_data);
                iter(1, t_data, cpst, temp_cpst);
            }
            temp_cpsts.insert(temp_cpsts.end(), temp_cpst.begin(), temp_cpst.end());

            rel_t_data.swap(t_data);
            rel_cpst.swap(temp_cpst);
        }

        std::vector<comparestruct>().swap(cpsts1);
        temp_cpsts.swap(cpsts);
        std::vector<comparestruct>().swap(temp_cpsts);
        malloc_trim(0);
    }
}