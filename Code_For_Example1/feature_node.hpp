#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <map>
#include <Eigen/Core>
#include <Eigen/SparseQR>
#include <eigen3/unsupported/Eigen/KroneckerProduct>
#include <sophus/se3.hpp>

class Feature_Frame;
class Feature_Id;

class FeatureFrameManager
{
private:
    std::map<uint, std::shared_ptr<Feature_Id>> features_;
    std::map<uint, Eigen::Vector2d> features_gt_;
    std::map<uint, std::shared_ptr<Feature_Frame>> frames_;
    std::map<uint, Eigen::Vector3d> frames_gt_;

public:
    FeatureFrameManager();
    void GetDataFromTXT(std::string path, uint frame_id);
    void GenDataFromTXT(std::string path, bool add_noise=true);
    void SaveGroundtruth(std::string path);
    void SaveMeasurement(std::string path, int bias=0);
    void MergeFrame();
    int GetFeatureSize();
    int GetFrameSize();
    std::vector<uint> GetFeaturesId();
    std::vector<uint> GetFramesId();
    void ComputeParams(Eigen::MatrixXd &A, Eigen::VectorXd &Z_xy, Eigen::VectorXd &Z_ori, Eigen::MatrixXd &C,
                       std::map<uint, Eigen::MatrixXd> &Ai, std::map<uint, Eigen::VectorXd> &Zi);
    void ComputeState(Eigen::MatrixXd &A, Eigen::VectorXd &Z, Eigen::MatrixXd &C, std::vector<double> &phis, std::string save_path);
    void ComputeState(Eigen::MatrixXd &A, Eigen::VectorXd &Z, Eigen::MatrixXd &C, std::vector<double> &phis, Eigen::VectorXd &X_);
    // Eigen::MatrixXd GetA();
    //~FeatureFrameManager();
private:
    void GetA(Eigen::MatrixXd &A,std::map<uint, Eigen::MatrixXd> &Ai);
    void GetZ_xy(Eigen::VectorXd &Z_xy, std::map<uint, Eigen::VectorXd> &Zi);
    void GetZ_ori(Eigen::VectorXd &Z_ori);
    void GetC(Eigen::MatrixXd &C);
    Eigen::MatrixXd GetR(int size, double phi);
};

class Feature_Frame
{
private:
    uint frame_id_;
    std::map<uint, std::weak_ptr<Feature_Id>> features_;
    Eigen::Matrix2d R_; // R_wc
    Eigen::Vector2d t_; // t_wc
    std::map<uint, Eigen::Vector2d> obs_f_;
    Eigen::Vector3d odom_;

public:
    Feature_Frame(int frame_id);
    Feature_Frame(int frame_id, Eigen::Vector3d odom);
    void AddFeature(std::shared_ptr<Feature_Id> feature, Eigen::Vector2d obs);
    void SetOdom(Eigen::Vector3d odom);
    void SetPose(Eigen::Matrix2d R, Eigen::Vector2d t);
    void reset();
    void MergeFrame(std::shared_ptr<Feature_Frame> other);
    uint GetFrameId();
    int GetMulitViewedFeatureNum();
    int GetSingleViewedFeatureNum();
    int GetFeatureNum();
    bool HasFeatureId(uint id);
    std::vector<std::weak_ptr<Feature_Id>> GetFeatures();
    std::vector<std::weak_ptr<Feature_Id>> GetMulitViewedFeatures();
    std::vector<std::weak_ptr<Feature_Id>> GetSingleViewedFeatures();
    std::vector<uint> GetFeaturesId();
    std::vector<uint> GetMulitViewedFeaturesId();
    std::vector<uint> GetSingleViewedFeaturesId();
    std::map<uint, Eigen::Vector2d> GetFeaturesObs();
    Eigen::Vector3d GetOdom();
    Eigen::Vector2d GetFeatureObs(uint feature_id);
    //~Feature_Frame();
};

class Feature_Id
{
private:
    uint feature_id_;
    Eigen::Vector2d xy_w_;
    std::map<uint, std::weak_ptr<Feature_Frame>> frames_;

public:
    Feature_Id(int _feature_id);
    void AddFrame(std::shared_ptr<Feature_Frame> frame);
    void reset();
    void SetWorldPostion(Eigen::Vector2d xy);
    uint GetFeatureId();
    int GetFrameNum();
    Eigen::Vector2d GetFeatureWorldPosition();
    std::vector<std::weak_ptr<Feature_Frame>> GetFrames();
    std::vector<uint> GetFramesId();
    //~Feature_Id();
};
