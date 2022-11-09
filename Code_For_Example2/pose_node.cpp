#include "pose_node.hpp"
#define pi 3.14159265358979323846

PoseNode::PoseNode(/* args */)
{
}

PoseNode::PoseNode(Eigen::Vector2d xy, double theta, int id) : posi_(xy), theta_(theta), id_(id)
{
}

PoseNode::PoseNode(double x, double y, double theta, int id) : theta_(theta), id_(id)
{
    posi_ = Eigen::Vector2d(x, y);
    if (theta_ > pi)
        theta_ -= 2 * pi;
    if (theta_ < -pi)
        theta_ += 2 * pi;
}
PoseNode::PoseNode(double x, double y, double theta, int id, int seed) : theta_(theta), id_(id), seed_(seed)
{
    posi_ = Eigen::Vector2d(x, y);
    if (theta_ > pi)
        theta_ -= 2 * pi;
    if (theta_ < -pi)
        theta_ += 2 * pi;
    setseed_ = true;
}

void PoseNode::AddParentNode(PoseNode::S_Ptr p_node)
{
    parent_nodes_[p_node->GetId()] = p_node;
    parent_size_++;
}

void PoseNode::AddChildNode(PoseNode::S_Ptr c_node)
{
    child_nodes_[c_node->GetId()] = c_node;
    if (setseed_)
        ComputeMea(c_node, seed_);
    else
        ComputeMea(c_node);
    has_child_ = true;
    child_size_++;
}

void PoseNode::ComputeMea(PoseNode::S_Ptr node)
{
    int run_time;
    std::ifstream runtimes_in;
    runtimes_in.close();
    std::string runtimes_dir = "../runtimes.txt";
    runtimes_in.open(runtimes_dir.c_str(), std::ios::in);
    runtimes_in >> run_time;
    runtimes_in.close();
    // std::cout << "run_time:" << run_time << std::endl;

    static std::default_random_engine e((time_t)run_time);
    static std::normal_distribution<double> inf_x(0, covariancen_(0, 0));
    static std::normal_distribution<double> inf_y(0, covariancen_(1, 1));
    static std::normal_distribution<double> inf_theta(0, covariancen_(2, 2));

    double noise_x = inf_x(e);
    double noise_y = inf_y(e);
    Eigen::Vector2d noise_xy(noise_x, noise_y);
    double noise_theta = inf_theta(e);

    auto pos_c = node->GetPosi();
    auto &pos_p = posi_;
    Sophus::SO2d so2_c = Sophus::SO2d::exp(node->Gettheta());
    Sophus::SO2d so2_p = Sophus::SO2d::exp(theta_);

    Eigen::Vector2d mea_post = so2_p.inverse() * (pos_c - pos_p);
    double mea_oir = (so2_p.inverse() * so2_c).log();

    if (add_noise_)
    {
        z_xy_[node->GetId()] = mea_post + noise_xy;
        z_theta_[node->GetId()] = mea_oir + noise_theta;
    }
    else
    {
        z_xy_[node->GetId()] = mea_post;
        z_theta_[node->GetId()] = mea_oir;
    }
    gt_z_xy_[node->GetId()] = mea_post;
    gt_z_theta_[node->GetId()] = mea_oir;
}

void PoseNode::ComputeMea(PoseNode::S_Ptr node, int seed)
{
    static std::default_random_engine e((time_t)seed);
    static std::normal_distribution<double> inf_x(0, covariancen_(0, 0));
    static std::normal_distribution<double> inf_y(0, covariancen_(1, 1));
    static std::normal_distribution<double> inf_theta(0, covariancen_(2, 2));

    double noise_x = inf_x(e);
    double noise_y = inf_y(e);
    Eigen::Vector2d noise_xy(noise_x, noise_y);
    double noise_theta = inf_theta(e);

    auto pos_c = node->GetPosi();
    auto &pos_p = posi_;
    Sophus::SO2d so2_c = Sophus::SO2d::exp(node->Gettheta());
    Sophus::SO2d so2_p = Sophus::SO2d::exp(theta_);

    Eigen::Vector2d mea_post = so2_p.inverse() * (pos_c - pos_p);
    double mea_oir = (so2_p.inverse() * so2_c).log();

    if (add_noise_)
    {
        z_xy_[node->GetId()] = mea_post + noise_xy;
        z_theta_[node->GetId()] = mea_oir + noise_theta;
    }
    else
    {
        z_xy_[node->GetId()] = mea_post;
        z_theta_[node->GetId()] = mea_oir;
    }
    gt_z_xy_[node->GetId()] = mea_post;
    gt_z_theta_[node->GetId()] = mea_oir;
}

void PoseNode::SetSeed(int seed)
{
    seed_ = seed;
    setseed_ = true;
}

Eigen::Vector2d PoseNode::GetPosi()
{
    return posi_;
}

double PoseNode::Gettheta()
{
    return theta_;
}

int PoseNode::GetId()
{
    return id_;
}

std::map<int, Eigen::Vector2d> PoseNode::GetZPosi()
{
    return gt_z_xy_;
}
Eigen::Vector2d PoseNode::GetZPosi(int id)
{
    return gt_z_xy_[id];
}
std::map<int, Eigen::Vector2d> PoseNode::GetZPosiWithNoise()
{
    return z_xy_;
}
Eigen::Vector2d PoseNode::GetZPosiWithNoise(int id)
{
    return z_xy_[id];
}

std::map<int, double> PoseNode::GetZtheta()
{
    return gt_z_theta_;
}
double PoseNode::GetZtheta(int id)
{
    return gt_z_theta_[id];
}
std::map<int, double> PoseNode::GetZthetaWithNoise()
{
    return z_theta_;
}
double PoseNode::GetZthetaWithNoise(int id)
{
    return z_theta_[id];
}

bool PoseNode::HasChild()
{
    return has_child_;
}

int PoseNode::GetChildSize()
{
    return child_size_;
}

int PoseNode::GetParentSize()
{
    return parent_size_;
}

std::map<int, PoseNode::S_Ptr> PoseNode::GetChilds()
{
    return child_nodes_;
}

void PoseNode::SetInformation(double x, double y, double theta, bool add_noise)
{
    covariancen_(0, 0) = x;
    covariancen_(1, 1) = y;
    covariancen_(2, 2) = theta;
    information_(0, 0) = 1. / (x * x);
    information_(1, 1) = 1. / (y * y);
    information_(2, 2) = 1. / (theta * theta);
    add_noise_ = add_noise;
}
void PoseNode::SetInformation(Eigen::Matrix3d &covariancen, bool add_noise)
{
    covariancen_ = covariancen;
    information_ = covariancen_.inverse() * covariancen_.inverse();
    add_noise_ = add_noise;
}
Eigen::Matrix3d PoseNode::GetInformation()
{
    return covariancen_;
}
