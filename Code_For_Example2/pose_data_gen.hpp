#ifndef POSE_DATA_GEN_H_
#define POSE_DATA_GEN_H_

#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <fstream>
#include <zconf.h>
#include <string>
#include <sstream>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <eigen3/unsupported/Eigen/KroneckerProduct>

#include <sophus/so2.hpp>

class PoseNode;

class PoseDataGen
{
public:
    PoseDataGen(/* args */);
    void AddPose(std::shared_ptr<PoseNode> node);
    void AddEdge(int p, int c);
    void SetInformation(double x, double y, double theta, bool add_noise = false);
    void PrintG2O(std::string path);
    std::map<int, std::shared_ptr<PoseNode>> GetNodes();

private:
    bool add_noise_;
    std::map<int, std::shared_ptr<PoseNode>> nodes_;
    std::vector<std::pair<int, int>> edges_;
    Eigen::Matrix3d information_;
    Eigen::Matrix3d covariancen_;
};

#endif