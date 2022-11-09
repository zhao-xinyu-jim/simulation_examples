#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <string>
#include <memory>
#include <algorithm>
#include <iomanip>
#include <omp.h>

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <yaml-cpp/yaml.h>
// #include "optimizer.hpp"
// #include "data_generator.hpp"
// #include "solver.hpp"
#include "pose_data_gen.hpp"
#include "pose_node.hpp"
#include "optimizer.hpp"
#include "tic_toc.hpp"

using namespace std;
using namespace Eigen;
using Sophus::SE3d;
using Sophus::SO3d;

#define pi 3.14159265358979323846

int main(int argc, char **argv)
{
    YAML::Node config_node = YAML::LoadFile("../config.yaml");
    double inf_x = config_node["inf_x"].as<double>();
    double inf_y = config_node["inf_y"].as<double>();
    double inf_th = config_node["inf_th"].as<double>();
    int step = config_node["step"].as<int>();
    int edge_size = config_node["edge_size"].as<int>();
    bool use_runtime_as_seed = config_node["use_runtime_as_seed"].as<bool>();
    std::string save_mea_as_G2O_path = config_node["save_mea_as_G2O_path"].as<std::string>();

    int id = 0;
    vector<shared_ptr<PoseNode>> r;
    if (use_runtime_as_seed)
        r.push_back(make_shared<PoseNode>(0, 0, 0, id++));
    else
        r.push_back(make_shared<PoseNode>(0, 0, 0, id++, time(0)));
    for (int i = 0; i < step; i++)
    {
        double x = config_node["x"][i].as<double>();
        double y = config_node["y"][i].as<double>();
        double theta = config_node["theta"][i].as<double>();
        if (use_runtime_as_seed)
            r.push_back(make_shared<PoseNode>(x, y, theta, id++));
        else
            r.push_back(make_shared<PoseNode>(x, y, theta, id++, time(0)));
    }

    shared_ptr<PoseDataGen> gen = make_shared<PoseDataGen>();
    gen->SetInformation(inf_x, inf_y, inf_th, true);
    for (int i = 0; i < step + 1; i++)
    {
        gen->AddPose(r[i]);
    }

    for (int i = 0; i < edge_size; i++)
    {
        int from = config_node["edge_from"][i].as<int>();
        int to = config_node["edge_to"][i].as<int>();
        gen->AddEdge(from, to);
    }

    shared_ptr<Optimizer> opt = make_shared<Optimizer>();
    auto nodes = gen->GetNodes();
    opt->AddNodes(nodes);
    opt->SetInformation(inf_x, inf_y, inf_th);
    opt->SetFixedId(0);
    opt->ComputeParam();

    int run_time;
    std::ifstream runtimes_in;
    std::ofstream runtimes_out;
    runtimes_in.close();
    runtimes_out.close();
    string runtimes_dir = "../runtimes.txt";
    runtimes_in.open(runtimes_dir.c_str(), std::ios::in);
    runtimes_in >> run_time;
    runtimes_in.close();
    runtimes_out.open(runtimes_dir.c_str(), std::ios::out);
    runtimes_out << (run_time + 1);
    runtimes_out.close();
    gen->PrintG2O(save_mea_as_G2O_path + std::to_string(run_time) + ".g2o");

    opt->solve();
}