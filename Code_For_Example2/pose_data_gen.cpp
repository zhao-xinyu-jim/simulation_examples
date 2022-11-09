#include "pose_node.hpp"
#include "pose_data_gen.hpp"
#include <float.h>

PoseDataGen::PoseDataGen(/* args */) : information_(Eigen::Matrix3d::Identity())
{
}

void PoseDataGen::AddPose(std::shared_ptr<PoseNode> node)
{
    node->SetInformation(covariancen_, add_noise_);
    nodes_[node->GetId()] = node;
}

void PoseDataGen::AddEdge(int p, int c)
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
    edges_.push_back(std::pair<int, int>(p, c));
    nodes_[p]->AddChildNode(nodes_[c]);
    nodes_[c]->AddParentNode(nodes_[p]);
}

void PoseDataGen::SetInformation(double x, double y, double theta, bool add_noise)
{
    covariancen_(0, 0) = x;
    covariancen_(1, 1) = y;
    covariancen_(2, 2) = theta;
    information_(0, 0) = 1. / (x * x);
    information_(1, 1) = 1. / (y * y);
    information_(2, 2) = 1. / (theta * theta);
    add_noise_ = add_noise;
}

void PoseDataGen::PrintG2O(std::string path)
{
    std::fstream fs(path, std::ios::out);
    for (auto &vertex : nodes_)
    {
        Eigen::Vector2d posi = vertex.second->GetPosi();
        double theta = vertex.second->Gettheta();
        fs << "VERTEX_SE2 " << vertex.first << " " << posi.x() << " " << posi.y() << " " << theta << std::endl;
    }
    for (auto &edge : edges_)
    {
        int p = edge.first;
        int c = edge.second;
        Eigen::Vector2d z_xy = nodes_[p]->GetZPosiWithNoise()[c];
        double z_theta = nodes_[p]->GetZthetaWithNoise()[c];
        fs << "EDGE_SE2 " << p << " " << c << " ";
        if (add_noise_)
            fs << z_xy.x() << " " << z_xy.y() << " " << z_theta;
        else
            fs << z_xy.x() << " " << z_xy.y() << " " << z_theta;

        for (size_t i = 0; i < 3; i++)
            for (size_t j = i; j < 3; j++)
                if (!add_noise_ && i == j)
                    fs << " " << 0.00001;
                    // fs << " " << DBL_MIN;
                else
                    fs << " " << information_(i, j);
        fs << std::endl;
    }
}

std::map<int, std::shared_ptr<PoseNode>> PoseDataGen::GetNodes()
{
    return nodes_;
}
