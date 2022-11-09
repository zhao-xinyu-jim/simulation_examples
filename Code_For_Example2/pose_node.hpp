#ifndef POSE_NODE_H_
#define POSE_NODE_H_

#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <fstream>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <sophus/so2.hpp>

class PoseNode
{
public:
    typedef std::shared_ptr<PoseNode> S_Ptr;
    typedef std::weak_ptr<PoseNode> W_Ptr;

public:
    PoseNode(/* args */);
    PoseNode(Eigen::Vector2d xy, double theta, int id);
    PoseNode(double x, double y, double theta, int id);
    PoseNode(double x, double y, double theta, int id ,int seed);
    void AddParentNode(PoseNode::S_Ptr node);
    void AddChildNode(PoseNode::S_Ptr node);
    void ComputeMea(PoseNode::S_Ptr node);
    void ComputeMea(PoseNode::S_Ptr node, int seed);
    void SetSeed(int seed);
    Eigen::Vector2d GetPosi();
    double Gettheta();
    int GetId();
    std::map<int, Eigen::Vector2d> GetZPosi();
    std::map<int, double> GetZtheta();
    Eigen::Vector2d GetZPosi(int id);
    double GetZtheta(int id);
    std::map<int, Eigen::Vector2d> GetZPosiWithNoise();
    std::map<int, double> GetZthetaWithNoise();
    Eigen::Vector2d GetZPosiWithNoise(int id);
    double GetZthetaWithNoise(int id);
    bool HasChild();
    int GetChildSize();
    int GetParentSize();
    std::map<int, S_Ptr> GetChilds();
    void SetInformation(double x, double y, double theta, bool add_noise = false);
    void SetInformation(Eigen::Matrix3d &covariancen, bool add_noise = false);
    Eigen::Matrix3d GetInformation();

private:
    std::map<int, S_Ptr> parent_nodes_;
    std::map<int, S_Ptr> child_nodes_;
    Eigen::Vector2d posi_;
    double theta_;
    int id_;
    bool has_child_ = false;
    bool add_noise_ = false;
    int child_size_ = 0;
    int parent_size_ = 0;
    int seed_ = -1;
    bool setseed_ = false;

    std::map<int, Eigen::Vector2d> z_xy_, gt_z_xy_;
    std::map<int, double> z_theta_, gt_z_theta_;

    Eigen::Matrix3d information_;
    Eigen::Matrix3d covariancen_;
};

#endif