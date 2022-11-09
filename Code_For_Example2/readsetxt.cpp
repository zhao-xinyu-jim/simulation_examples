#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <string>
#include <memory>
#include <algorithm>
#include <iomanip>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include "pose_node.hpp"
#include "optimizer.hpp"
#include <yaml-cpp/yaml.h>

using namespace std;

std::map<int, std::shared_ptr<PoseNode>> nodes_;
#define pi 3.14159265358979323846

void AddPose(std::shared_ptr<PoseNode> &node, double inf_x, double inf_y, double inf_th)
{
    node->SetInformation(inf_x, inf_y, inf_th, true);
    nodes_[node->GetId()] = node;
}

void AddEdge(int p, int c)
{
    if (nodes_.find(p) == nodes_.end())
    {
        std::cout << "node " << p << "has not been registered" << std::endl;
        return;
    }
    if (nodes_.find(c) == nodes_.end())
    {
        std::cout << "node " << c << "has not been registered" << std::endl;
        return;
    }
    nodes_[p]->AddChildNode(nodes_[c]);
    nodes_[c]->AddParentNode(nodes_[p]);
}
void AddEdge(std::shared_ptr<PoseNode> &p, std::shared_ptr<PoseNode> &c)
{
    p->AddChildNode(c);
    c->AddParentNode(p);
}

int main(int argc, char **argv)
{
    YAML::Node config_node = YAML::LoadFile("../config.yaml");
    double inf_x = config_node["inf_x"].as<double>();
    double inf_y = config_node["inf_y"].as<double>();
    double inf_th = config_node["inf_th"].as<double>();

    int run_time;
    string runtimes_dir = "../runtimes.txt";
    std::ifstream runtimes_in;
    runtimes_in.close();
    runtimes_in.open(runtimes_dir.c_str(), std::ios::in);
    runtimes_in >> run_time;
    runtimes_in.close();

    ifstream fin;
    fin.close();
    fin.open("../data1/temp/" + std::to_string(run_time - 1) + ".txt", std::ios::in);
    string line_info, input_result;
    std::vector<std::vector<double>> vec;
    for (int i = 0; getline(fin, line_info); i++) // line中不包括每行的换行符
    {
        stringstream input(line_info);
        std::vector<double> temp_vec;
        //依次输出到input_result中，并存入Eigen::MatrixXd points中
        for (int j = 0; input >> input_result; ++j)
        {
            string::size_type size;
            temp_vec.push_back(stod(input_result, &size)); // string 转float
        }
        vec.push_back(temp_vec);
    }
    fin.close();

    int num_poses = 0;
    ifstream g2o;
    g2o.close();
    g2o.open("../data1/temp/" + std::to_string(run_time - 1) + ".g2o", std::ios::in);
    for (int i = 0; getline(g2o, line_info); i++) // line中不包括每行的换行符
    {
        stringstream input(line_info);
        input >> input_result;
        if (input_result == "VERTEX_SE2" || input_result == "VERTEX_SE3")
            num_poses++;
    }
    g2o.close();

    int rows, cols;
    rows = vec.size();
    cols = vec[0].size();
    Eigen::MatrixXd X_(rows, cols);
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            X_(i, j) = vec[i][j];
    // cout << X_ << endl;

    int dim = X_.rows();
    int x_cols = X_.cols();
    cout << "dim:" << dim << endl;
    cout << "x_cols:" << x_cols << endl;
    cout << endl;
    vector<Eigen::MatrixXd> R_s(num_poses);
    vector<Eigen::VectorXd> t_s(num_poses);

    if (dim == 3)
        for (size_t i = 0; i < num_poses; i++)
        {
            R_s[i] = Eigen::Matrix3d::Identity();
            t_s[i] = Eigen::Vector3d::Zero();
        }
    else if (dim == 2)
        for (size_t i = 0; i < num_poses; i++)
        {
            R_s[i] = Eigen::Matrix2d::Identity();
            t_s[i] = Eigen::Vector2d::Zero();
        }

    for (size_t i = 0; i < num_poses; i++)
    {
        R_s[i] = X_.block(0, x_cols - dim * num_poses + i * dim, dim, dim);
        t_s[i] = X_.block(0, i, dim, 1);
        // cout << R_s[i] << endl;
        // cout << endl;
    }
    Eigen::MatrixXd R0 = R_s[0];
    for (size_t i = 0; i < num_poses; i++)
    {
        Eigen::MatrixXd R = R_s[i];
        Eigen::VectorXd t = t_s[i];

        R_s[i] = R0.transpose() * R;
        t_s[i] = R0.transpose() * t;
        // cout << R_s[i] << endl;
        cout << t_s[i].transpose() << endl;
        // cout << endl;
    }
    std::map<int, double> phis;

    ofstream out;
    out.close();
    // out.open("../data/f_bound/" + std::to_string(run_time - 1) + "_sesyn.txt", std::ios::out);
    // out.open("../data1/f_bound/1_sesyn.txt", std::ios::app);
    for (size_t i = 0; i < num_poses; i++)
    {
        Eigen::Matrix2d R = R_s[i];
        Sophus::SO2d so2 = Sophus::SO2d::fitToSO2(R);
        cout << so2.log() << endl;
        // out << so2.log() << ",";
        phis[i] = so2.log();
    }
    // out.close();

    int id = 0;
    // shared_ptr<PoseNode> r0 = make_shared<PoseNode>(0, 0, 0, id++, run_time - 1); // x,y,theta,id
    // shared_ptr<PoseNode> r1 = make_shared<PoseNode>(100, 0, 1.2, id++, run_time - 1);
    // shared_ptr<PoseNode> r2 = make_shared<PoseNode>(200, 50, 2.5, id++, run_time - 1);
    // shared_ptr<PoseNode> r3 = make_shared<PoseNode>(100, 100, -2.5, id++, run_time - 1);
    // shared_ptr<PoseNode> r4 = make_shared<PoseNode>(0, 50, -1.2, id++, run_time - 1);
    // shared_ptr<PoseNode> r0 = make_shared<PoseNode>(0, 0, 0, id++, run_time - 1); // x,y,theta,id
    // shared_ptr<PoseNode> r1 = make_shared<PoseNode>(100, 0, -pi*0.8, id++, run_time - 1);
    // shared_ptr<PoseNode> r2 = make_shared<PoseNode>(19, -59, pi*0.4, id++, run_time - 1);
    // shared_ptr<PoseNode> r3 = make_shared<PoseNode>(50, 29, -pi*0.4, id++, run_time - 1);
    // shared_ptr<PoseNode> r4 = make_shared<PoseNode>(81, -59, pi*0.8, id++, run_time - 1);
    shared_ptr<PoseNode> r0 = make_shared<PoseNode>(0, 0, 0, id++, run_time - 1); // x,y,theta,id
    shared_ptr<PoseNode> r1 = make_shared<PoseNode>(100, 0, pi/2., id++, run_time - 1);
    shared_ptr<PoseNode> r2 = make_shared<PoseNode>(100, 100, pi, id++, run_time - 1);
    shared_ptr<PoseNode> r3 = make_shared<PoseNode>(0, 100, -pi/2., id++, run_time - 1);
    shared_ptr<PoseNode> r4 = make_shared<PoseNode>(0, 0, 0, id++, run_time - 1);
    AddPose(r0, inf_x, inf_y, inf_th);
    AddPose(r1, inf_x, inf_y, inf_th);
    AddPose(r2, inf_x, inf_y, inf_th);
    AddPose(r3, inf_x, inf_y, inf_th);
    AddPose(r4, inf_x, inf_y, inf_th);
    AddEdge(0, 1);
    AddEdge(1, 2);
    AddEdge(2, 3);
    AddEdge(3, 4);
    AddEdge(4, 0);
    AddEdge(0, 3);
    AddEdge(4, 1);

    // AddEdge(0, 2);
    // AddEdge(4, 2);
    // AddEdge(3, 1);

    shared_ptr<Optimizer> opt = make_shared<Optimizer>();
    opt->AddNodes(nodes_);
    opt->SetInformation(inf_x, inf_y, inf_th);
    opt->SetFixedId(0);
    opt->ComputeParam();
    out.open("../data1/f_bound/6_sesyn.txt", std::ios::app);
    cout << "f:" << opt->ObjFunc(phis) << endl;
    out << opt->ObjFunc(phis) << std::endl;
    out.close();
}