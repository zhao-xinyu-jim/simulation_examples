#include "optimizer.hpp"
#include "pose_node.hpp"
#include <cfloat>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include "tic_toc.hpp"
#define pi 3.14159265358979323846

bool compare1(comparestruct &a, comparestruct &b)
{
    return a.cost < b.cost;
}
bool compare2(comparestruct &a, comparestruct &b)
{
    return a.upper_bound < b.upper_bound;
}

Eigen::Matrix2d toRoation(double phi)
{
    Eigen::Matrix2d R;
    R << cos(phi), -sin(phi), sin(phi), cos(phi);
    return R;
}

Optimizer::Optimizer(/* args */)
{
}
void Optimizer::SetInformation(double x, double y, double theta)
{
    inf_x_ = 1. / (x * x);
    inf_y_ = 1. / (y * y);
    inf_theta_ = 4 * 1. / (theta * theta);
    // std::cout << inf_theta_ << std::endl;
}

void Optimizer::SetFixedId(int id)
{
    fixed_id_ = id;
}

void Optimizer::AddNodes(std::map<int, std::shared_ptr<PoseNode>> &nodes)
{
    nodes_ = nodes;
}

void Optimizer::ComputeParam()
{
    std::map<int, int> table;
    int A_rows = nodes_.size() - 1;
    int A_cols = 0;
    int rows_counter = 0;
    for (auto &node : nodes_)
    {
        int id = node.first;
        int cols = node.second->GetChildSize();
        A_cols += cols;
        if (id == fixed_id_)
            continue;

        table[id] = rows_counter;
        rows_counter++;
    }

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(A_cols * 2, A_cols * 2);
    for (int i = 0; i < A_cols; i++)
    {
        C(2 * i, 2 * i) = inf_x_;
        C(2 * i + 1, 2 * i + 1) = inf_y_;
    }
    // std::cout << C << std::endl
    //           << std::endl;

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(A_rows * 2, A_cols * 2);
    std::map<int, Eigen::MatrixXd> Ai;
    Eigen::VectorXd z = Eigen::MatrixXd::Zero(A_cols * 2, 1);
    std::map<int, Eigen::VectorXd> zi_perp;
    std::map<int, Eigen::VectorXd> zi;
    std::map<int, Eigen::MatrixXd> Qi;

    int cols_counter = 0;
    for (auto &node : nodes_)
    {
        int id = node.first;
        // int cols = node.second->GetChildSize();
        Ai[id] = Eigen::MatrixXd::Zero(A_rows * 2, A_cols * 2);
        zi[id] = Eigen::MatrixXd::Zero(A_cols * 2, 1);
        zi_perp[id] = Eigen::MatrixXd::Zero(A_cols * 2, 1);
        for (auto &child_node : node.second->GetChilds())
        {
            int child_id = child_node.first;
            if (table.find(child_id) != table.end())
            {
                Ai[id].block(table[child_id] * 2, cols_counter * 2, 2, 2) = Eigen::Matrix2d::Identity();
                A.block(table[child_id] * 2, cols_counter * 2, 2, 2) = Eigen::Matrix2d::Identity();
            }
            if (table.find(id) != table.end())
            {
                Ai[id].block(table[id] * 2, cols_counter * 2, 2, 2) = -Eigen::Matrix2d::Identity();
                A.block(table[id] * 2, cols_counter * 2, 2, 2) = -Eigen::Matrix2d::Identity();
            }
            zi[id].block(cols_counter * 2, 0, 2, 1) = node.second->GetZPosiWithNoise(child_id);
            zi_perp[id].block(cols_counter * 2, 0, 2, 1) = toRoation(pi / 2.) * node.second->GetZPosiWithNoise(child_id);
            z.block(cols_counter * 2, 0, 2, 1) = node.second->GetZPosiWithNoise(child_id);

            cols_counter++;
        }
        // std::cout << Ai[id] << std::endl
        //           << std::endl;
        // std::cout << zi[id].transpose() << std::endl
        //           << std::endl;
    }
    // std::cout << z.transpose() << std::endl
    //           << std::endl;

    Eigen::SparseMatrix<double> A_spa = A.sparseView();
    Eigen::SparseMatrix<double> C_spa = C.sparseView();
    Eigen::SparseMatrix<double> z_spa = z.sparseView();
    Eigen::SparseMatrix<double> ACAT = (A_spa * C_spa * A_spa.transpose());
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(ACAT.rows(), ACAT.cols());
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::AMDOrdering<int>> QRsolver;
    QRsolver.compute(ACAT);
    Eigen::SparseMatrix<double> ACAT_inv = QRsolver.solve(I.sparseView());
    for (auto &node : nodes_)
    {
        int id = node.first;
        Eigen::SparseMatrix<double> Ai_spa = Ai[id].sparseView();
        Eigen::SparseMatrix<double> Qi_spa = C_spa * Ai_spa.transpose() * ACAT_inv * A_spa * C_spa;
        Qi[id] = Qi_spa.toDense();
        // std::cout << Qi[id] << std::endl
        //           << std::endl;
    }

    ACA_inv_AC_spa_ = ACAT_inv * A_spa * C_spa;
    z_spa_ = z.sparseView();

    bij_ = Eigen::MatrixXd::Zero(nodes_.size(), nodes_.size());
    phiij0_ = Eigen::MatrixXd::Zero(nodes_.size(), nodes_.size());
    int counti = 0, countj = 0;
    for (auto &node1 : nodes_)
    {
        int i = node1.first;
        countj = 0;
        for (auto &node2 : nodes_)
        {
            int j = node2.first;
            bij_(counti, countj) = sqrt(pow(z.transpose() * Qi[i] * zi[j], 2) + pow(z.transpose() * Qi[i] * zi_perp[j], 2));
            phiij0_(counti, countj) = atan2(z.transpose() * Qi[i] * zi_perp[j], z.transpose() * Qi[i] * zi[j]);
            countj++;
        }
        counti++;
    }

    c1_ = (z_spa.transpose() * C_spa * z_spa).toDense()(0, 0);

    for (size_t i = 0; i < nodes_.size(); i++)
        c1_ -= bij_(i, i);
    c1_ += A_cols * inf_theta_;

    betaij0_ = Eigen::MatrixXd::Zero(nodes_.size(), nodes_.size());
    philowij0_ = Eigen::MatrixXd::Zero(nodes_.size(), nodes_.size());
    counti = 0, countj = 0;
    for (auto &node1 : nodes_)
    {
        int i = node1.first;
        auto childs = node1.second->GetChilds();
        countj = 0;
        for (auto &node2 : nodes_)
        {
            int j = node2.first;
            if (i == j)
            {
                countj++;
                continue;
            }
            if (childs.find(j) != childs.end())
            {
                double bij_2 = bij_(counti, countj) * bij_(counti, countj);
                double gamma_2 = inf_theta_ * inf_theta_;
                double complex = 2 * bij_(counti, countj) * inf_theta_ * cos(phiij0_(counti, countj) - node1.second->GetZthetaWithNoise(j));
                betaij0_(counti, countj) = sqrt(bij_2 + gamma_2 + complex);
                philowij0_(counti, countj) = atan2(bij_(counti, countj) * sin(phiij0_(counti, countj)) + inf_theta_ * sin(node1.second->GetZthetaWithNoise(j)), bij_(counti, countj) * cos(phiij0_(counti, countj)) + inf_theta_ * cos(node1.second->GetZthetaWithNoise(j)));
            }
            else
            {
                betaij0_(counti, countj) = bij_(counti, countj);
                philowij0_(counti, countj) = phiij0_(counti, countj);
            }

            countj++;
        }
        counti++;
    }
}

double Optimizer::ObjFunc(std::map<int, double> &phis)
{
    int counti = 0, countj = 0;
    double f = c1_;
    for (auto &node1 : nodes_)
    {
        int i = node1.first;
        countj = 0;
        for (auto &node2 : nodes_)
        {
            int j = node2.first;
            if (i == j)
            {
                countj++;
                continue;
            }
            f -= betaij0_(counti, countj) * cos(phis[j] - phis[i] - philowij0_(counti, countj));
            countj++;
        }
        counti++;
    }
    return f;
}

int Optimizer::CompareRange(double &max1, double &min1, double &max2, double &min2)
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

void Optimizer::MaxMinCos(double phi_0, double phi_1, double &max, double &min)
{
    // Sophus::SO2d so2_0 = Sophus::SO2d::exp(phi_0);
    // Sophus::SO2d so2_1 = Sophus::SO2d::exp(phi_1);
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

void Optimizer::ComputeBound(comparestruct &phis)
{
    phis.lower_bound = c1_;
    phis.upper_bound = c1_;
    std::deque<double> &phi1 = phis.a1;
    std::deque<double> &phi2 = phis.a2;

    int counti = 0, countj = 0;
    for (auto &node1 : nodes_)
    {
        int i = node1.first;
        countj = 0;
        for (auto &node2 : nodes_)
        {
            int j = node2.first;
            if (i == j)
            {
                countj++;
                continue;
            }
            double max = 1, min = -1;
            double phi_1 = phi1[countj] - phi2[counti] - philowij0_(counti, countj);
            double phi_2 = phi2[countj] - phi1[counti] - philowij0_(counti, countj);
            MaxMinCos(phi_1, phi_2, max, min);
            phis.upper_bound -= betaij0_(counti, countj) * min;
            phis.lower_bound -= betaij0_(counti, countj) * max;
            countj++;
        }
        counti++;
    }

    // std::cout << std::endl;
    // std::cout << "low:" << phis.lower_bound << std::endl;
    // std::cout << "up:" << phis.upper_bound << std::endl;
    // std::cout << std::endl;
}
void Optimizer::iter1(int layer, std::vector<std::vector<double>> &data, std::vector<double> &temp, std::vector<std::vector<double>> &results)
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
void Optimizer::ComputeBound1(comparestruct &phis, int divide_size)
{
    phis.lower_bound = c1_;
    phis.upper_bound = c1_;
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

        for (int i = 0; i < nodes_.size(); i++)
        {
            for (int j = 0; j < nodes_.size(); j++)
            {
                if (i == j)
                    continue;
                double c = cos(result[j] - result[i] - philowij0_(i, j));

                current -= betaij0_(i, j) * c;
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

void Optimizer::iter(int layer, std::vector<std::vector<double>> &data, comparestruct &cpst, std::vector<comparestruct> &cpsts)
{
    int max_layer = data.size() - 1;
    for (size_t i = 0; i < data[layer].size() - 1; i++)
    {
        if (layer == max_layer)
        {
            cpst.a1.push_back(data[layer][i]);
            cpst.a2.push_back(data[layer][i + 1]);
            cpsts.push_back(cpst);

            cpst.a1.pop_back();
            cpst.a2.pop_back();
        }
        else
        {
            cpst.a1.push_back(data[layer][i]);
            cpst.a2.push_back(data[layer][i + 1]);
            iter(layer + 1, data, cpst, cpsts);
            cpst.a1.pop_back();
            cpst.a2.pop_back();
        }
    }
}

void Optimizer::GeneratePhi(double begin, double end, int devide_size, std::vector<std::vector<double>> &data)
{
    data.resize(nodes_.size());
    data[0] = std::vector<double>(devide_size + 1, 0);
    int counti = 0, countj = 0;
    for (auto &node1 : nodes_)
    {
        if (counti == 0)
        {
            counti++;
            continue;
        }
        data[counti].clear();
        data[counti].resize(devide_size + 1);

        for (int countj = 0; countj <= devide_size; countj++)
        {
            data[counti][countj] = (end - begin) * ((double)countj / (double)devide_size) + begin;
            // std::cout << "data[" << counti << "][" << countj << "]:" << data[counti][countj] << " ";
        }
        // std::cout << std::endl;
        counti++;
    }
}

void Optimizer::GeneratePhi(std::deque<double> &begin, std::deque<double> &end, int devide_size, std::vector<std::vector<double>> &data)
{
    data.clear();
    data.resize(nodes_.size());
    data[0] = std::vector<double>(devide_size + 1, 0);
    int counti = 0, countj = 0;
    for (auto &node1 : nodes_)
    {
        if (counti == 0)
        {
            counti++;
            continue;
        }
        data[counti].clear();
        data[counti].resize(devide_size + 1);
        for (int countj = 0; countj <= devide_size; countj++)
        {
            data[counti][countj] = (end[counti] - begin[counti]) * ((double)countj / (double)devide_size) + begin[counti];
            // std::cout << "data[i][j]:" << data[counti][countj] << " ";
        }
        // std::cout << std::endl;
        counti++;
    }
}

Eigen::MatrixXd Optimizer::ComputeX(std::map<int, double> &phis)
{
    int A_rows = nodes_.size() - 1;
    int A_cols = 0;
    for (auto &node : nodes_)
    {
        int id = node.first;
        int cols = node.second->GetChildSize();
        A_cols += cols;
        if (id == fixed_id_)
            continue;
    }
    // std::cout << A_rows << std::endl;
    // std::cout << A_cols << std::endl;
    Eigen::MatrixXd R_hat = Eigen::MatrixXd::Zero(A_cols * 2, A_cols * 2);
    int cols_counter = 0;
    for (auto &node : nodes_)
    {
        int id = node.first;
        Eigen::Matrix2d R_phi = toRoation(phis[id]);
        // int cols = node.second->GetChildSize();
        for (auto &child_node : node.second->GetChilds())
        {
            int child_id = child_node.first;

            R_hat.block(cols_counter * 2, cols_counter * 2, 2, 2) = R_phi;
            cols_counter++;
        }
    }
    Eigen::SparseMatrix<double> R_hat_spa = R_hat.sparseView();
    Eigen::SparseMatrix<double> X_spa = ACA_inv_AC_spa_ * R_hat_spa * z_spa_;
    Eigen::MatrixXd X = X_spa.toDense();
    // std::cout << X.transpose() << std::endl;
    return X.transpose();
}

std::map<int, double> Optimizer::solve()
{
    YAML::Node config_node = YAML::LoadFile("../config.yaml");
    int iterations = config_node["iter_times"].as<int>();
    int start_random = config_node["start_frame_for_random"].as<int>();
    int threads_num = config_node["threads"].as<int>();
    int divide_size = config_node["divide_size"].as<int>();
    double eps = config_node["eps"].as<double>();
    std::string path = config_node["save_results_path"].as<std::string>();

    TicToc tic_toc;
    std::vector<std::vector<double>> data, rel_data;
    comparestruct cpst;
    std::vector<comparestruct> cpsts, cpsts1, temp_cpsts;
    GeneratePhi(-pi, pi, 2, data);
    iter(1, data, cpst, cpsts);
    rel_data.swap(data);
    // std::cout << "cpsts.size:" << cpsts.size() << std::endl;
    // std::cout << std::endl;

    std::ofstream a1, a2;
    std::ofstream opt_range, true_range;
    std::ofstream time_range;

    std::string f_range_dir = path + "/f_range.txt";
    std::string t_range_dir = path + "/t_range.txt";
    std::string x_dir = path + "/x.txt";
    for (int k = 0; k < iterations; k++)
    {
        std::cout << "==========iter:" << k << "=========="
                  << "\n";
        std::string k_string = std::to_string(k);
        tic_toc.tic();
        if (k < iterations - 1)
        {
            int countss = 0;
            omp_set_num_threads(threads_num);
#pragma omp parallel
            {
#pragma omp for
                for (int i = 0; i < cpsts.size(); i++)
                {
                    auto &cp = cpsts[i];
                    cp.a1.push_front(0);
                    cp.a2.push_front(0);
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

        comparestruct &min_cp = *min_element(cpsts.begin(), cpsts.end(), compare2);
        double min_max = min_cp.upper_bound;
        double min_min = min_cp.lower_bound;
        bool if_end = false;
        if (min_max - min_min < eps)
            if_end = true;

        if (k == iterations - 1 || if_end)
        {
            for (auto &cp : cpsts)
            {
                std::map<int, double> phis;
                if (k == iterations - 1)
                {
                    for (size_t i = 0; i < cp.a1.size(); i++)
                    {
                        double aa = (cp.a1[i] + cp.a2[i]) / 2.;
                        phis[i + 1] = aa;
                    }
                    phis[0] = 0;
                }
                else
                {
                    for (size_t i = 0; i < cp.a1.size(); i++)
                    {
                        double aa = (cp.a1[i] + cp.a2[i]) / 2.;
                        phis[i] = aa;
                    }
                }
                cp.cost = ObjFunc(phis);
            }
            comparestruct &cp = *min_element(cpsts.begin(), cpsts.end(), compare1);
            if (k == iterations - 1)
            {
                cp.a1.push_front(0);
                cp.a2.push_front(0);
            }
            ComputeBound1(cp, divide_size);
            std::cout << std::setprecision(21) << "lower_bound:" << cp.lower_bound << " upper_bound:" << cp.upper_bound << std::endl;

            // std::string a1_dir = path + "/" + file_name + "_a1.txt";
            // std::string a2_dir = path + "/" + file_name + "_a2.txt";
            // a1.open(a1_dir.c_str(), std::ios::app);
            // a2.open(a2_dir.c_str(), std::ios::app);
            // for (size_t i = 0; i < cp.a1.size(); i++)
            // {
            //     a1 << cp.a1[i] << " ";
            //     a2 << cp.a2[i] << " ";
            // }
            // a1 << std::endl;
            // a2 << std::endl;
            // a1.close();
            // a2.close();

            opt_range.open(f_range_dir.c_str(), std::ios::app);
            double cost = cp.cost;
            std::cout << "final_cost:" << cost << std::endl;
            opt_range << cost << std::endl;
            opt_range.close();
            time_range.open(t_range_dir.c_str(), std::ios::app);
            time_range << std::endl;
            time_range.close();
            std::map<int, double> phis;
            for (size_t i = 0; i < cp.a1.size(); i++)
            {
                phis[i]= (cp.a1[i] + cp.a2[i]) / 2.;
            }
            true_range.open(x_dir.c_str(), std::ios::app);
            true_range << ComputeX(phis) << std::endl
                       << std::endl;
            true_range.close();

            return phis;
        }
        time_range.open(t_range_dir.c_str(), std::ios::app);
        double timecost = tic_toc.toc();
        time_range << timecost << " ";
        time_range.close();
        std::cout << "time_cost: " << timecost << std::endl;
        int count_add = 0;
        int count_earse = 0;

        std::vector<comparestruct>().swap(temp_cpsts);
        std::cout << "min_max:" << min_max << "\n";
        std::cout << "min_min:" << min_min << std::endl;

        // std::cout << "begin add and earse" << std::endl;
        for (auto iter = cpsts.begin(); iter != cpsts.end(); iter++)
        {
            bool isnearby = true;
            if (iter->lower_bound > min_max || !isnearby) //|| dis > c_cur.size() * cubeedgelength
                count_earse++;
            else
            {
                cpsts1.push_back(*iter);
                count_add++;
            }
        }
        std::vector<comparestruct>().swap(cpsts);

        std::cout << "count_add:" << count_add << "\n";
        std::cout << "count_ear:" << count_earse << "\n";
        std::cout << "end add and earse"
                  << "\n";
        std::cout << "cpsts1.size:" << cpsts1.size() << "\n";
        {
            temp_cpsts.clear();
            for (auto &cp : cpsts1)
            {
                std::vector<comparestruct> temp_cpst;
                std::vector<std::vector<double>> t_data;

                if (k < iterations)
                {
                    GeneratePhi(cp.a1, cp.a2, 2, t_data);
                    iter(1, t_data, cpst, temp_cpst);
                }
                temp_cpsts.insert(temp_cpsts.end(), temp_cpst.begin(), temp_cpst.end());

                std::vector<std::vector<double>>().swap(t_data);
                std::vector<comparestruct>().swap(temp_cpst);
            }
        }

        std::vector<comparestruct>().swap(cpsts1);
        temp_cpsts.swap(cpsts);
        std::vector<comparestruct>().swap(temp_cpsts);

        std::cout << "cpsts.size():" << cpsts.size() << std::endl;
    }
}