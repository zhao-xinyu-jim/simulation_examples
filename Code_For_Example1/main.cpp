#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <string>
#include <malloc.h>
#include <memory>
#include <algorithm>
#include <iomanip>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <yaml-cpp/yaml.h>

#include "feature_node.hpp"
#include "solver.hpp"
#include "tic_toc.hpp"

using namespace std;
using namespace Eigen;
using Sophus::SE3d;
using Sophus::SO3d;

#define pi 3.14159265358979323846

int times = 0;
int main(int argc, char **argv)
{
    YAML::Node config_node = YAML::LoadFile("../config.yaml");
    int iter_times = config_node["iter_times"].as<int>();
    int step_size = config_node["step_size"].as<int>();
    double gama = config_node["gama"].as<double>();
    std::string save_time_path = config_node["path"].as<std::string>();
    std::string save_mea_path = config_node["save_mea_path"].as<std::string>();
    std::string save_gt_path = config_node["save_gt_path"].as<std::string>();
    std::string get_txt_path = config_node["get_txt_path"].as<std::string>();

    std::shared_ptr<FeatureFrameManager> fm = std::make_shared<FeatureFrameManager>();
    fm->GenDataFromTXT(get_txt_path, true);
    fm->SaveGroundtruth(save_gt_path);
    fm->SaveMeasurement(save_mea_path, i * 5);

    if (fm->GetFeatureSize() == 0 || fm->GetFrameSize() == 0)
    {
        assert("fm->GetFeatureSize() == 0 || fm->GetFrameSize() == 0 !");
        return 0;
    }

    Eigen::MatrixXd A;
    Eigen::VectorXd Z_xy;
    Eigen::VectorXd Z_ori;
    Eigen::MatrixXd C;
    std::map<uint, Eigen::MatrixXd> Ai;
    std::map<uint, Eigen::VectorXd> Zi;
    fm->ComputeParams(A, Z_xy, Z_ori, C, Ai, Zi);

    std::shared_ptr<Solver> solver_ptr = std::make_shared<Solver>(feature_size, pose_size);
    solver_ptr->SetA(A);
    solver_ptr->SetAi(Ai);
    solver_ptr->SetZ(Z_xy, Z_ori);
    solver_ptr->SetZi(Zi);
    solver_ptr->SetC(C);
    solver_ptr->SetGama(gama);

    std::map<uint, Eigen::MatrixXd>().swap(Ai);
    std::map<uint, Eigen::VectorXd>().swap(Zi);
    malloc_trim(0);

    solver_ptr->ComputeParam();
    solver_ptr->release();

    auto phis = solver_ptr->solve(iter_times);
    std::ofstream file_time;
    std::cout << "phis:";
    file_time.open(save_time_path.c_str(), std::ios::app);
    for (size_t i = 0; i < phis.size(); i++)
    {
        std::cout << phis[i] << " ";
        file_time << phis[i] << " ";
    }
    std::cout << endl;
    file_time << endl;
    Eigen::VectorXd X_;
    fm->ComputeState(A, Z_xy, C, phis, X_);
    std::cout << X_.transpose() << endl;
    file_time << X_.transpose() << endl;
    file_time << endl;
    file_time.close();

    return 0;
}