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
    std::string read_submap_datapath = config_node["read_submap_datapath"].as<std::string>();
    std::string save_submap_datapath = config_node["save_submap_datapath"].as<std::string>();
    std::string save_path = save_submap_datapath + std::to_string(step_size) + "step";

    size_t end = 5;
    for (size_t count = 1; count < end + 1; count--)
    {
        times++;
        std::shared_ptr<FeatureFrameManager> fm = std::make_shared<FeatureFrameManager>();

        for (size_t i = 0; i < step_size + 1; i++)
        {
            if (count > end)
                break;
            std::string path = read_submap_datapath + std::to_string(count) + ".txt";
            std::cout << path << std::endl;

            fm->GetDataFromTXT(path, i);
            count++;
        }
        auto feas = fm->GetFeaturesId();
        auto fras = fm->GetFramesId();
        if (fm->GetFrameSize() == 0 || fm->GetFeatureSize() == 0)
        {
            break;
        }

        Eigen::MatrixXd A;
        Eigen::VectorXd Z_xy;
        Eigen::VectorXd Z_ori;
        Eigen::MatrixXd C;
        std::map<uint, Eigen::MatrixXd> Ai;
        std::map<uint, Eigen::VectorXd> Zi;
        fm->ComputeParams(A, Z_xy, Z_ori, C, Ai, Zi);

        int feature_size = fm->GetFeatureSize();
        int pose_size = fm->GetFrameSize();
        std::shared_ptr<Solver> solver_ptr = std::make_shared<Solver>(feature_size, pose_size);
        solver_ptr->SetA(A);
        solver_ptr->SetAi(Ai);
        solver_ptr->SetZ(Z_xy, Z_ori);
        solver_ptr->SetZi(Zi);
        solver_ptr->SetC(C);
        solver_ptr->SetGama(gama);

        std::map<uint, Eigen::MatrixXd>().swap(Ai);
        Z_ori = Eigen::VectorXd();
        std::map<uint, Eigen::VectorXd>().swap(Zi);
        malloc_trim(0);

        solver_ptr->ComputeParam();
        solver_ptr->release();

        auto phis = solver_ptr->solve(iter_times);

        std::cout << "phis:";
        for (size_t i = 0; i < phis.size(); i++)
        {
            std::cout << phis[i] << " ";
        }
        std::cout << endl;

        Eigen::VectorXd X_;
        fm->ComputeState(A, Z_xy, C, phis, save_path + "_" + std::to_string(times) + ".txt");
        std::cout << endl;
    }
    return 0;
}