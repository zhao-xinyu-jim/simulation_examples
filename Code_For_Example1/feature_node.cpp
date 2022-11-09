#include "feature_node.hpp"
#include <string>
#include <malloc.h>
#include <iomanip>

Eigen::Matrix2d R(double phi)
{
    Eigen::Matrix2d r;
    r << cos(phi), -sin(phi), sin(phi), cos(phi);
    return r;
}

FeatureFrameManager::FeatureFrameManager()
{
}
void FeatureFrameManager::GetDataFromTXT(std::string path, uint frame_id)
{
    std::ifstream f;
    f.close();
    f.open(path, std::ios::in);
    if (!f.is_open())
    {
        return;
    }
    std::string line;
    uint feature_id;
    Eigen::Vector3d odom;
    Eigen::Vector2d obs;
    for (int i = 0; getline(f, line); i++) // line中不包括每行的换行符
    {
        std::stringstream input(line);
        std::string temp_data;
        if (i < 3)
        {
            input >> temp_data;
            input >> temp_data;
            odom(i) = std::stod(temp_data);
            if (i == 2 && frames_.find(frame_id) == frames_.end())
                frames_[frame_id] = std::make_shared<Feature_Frame>(frame_id, odom);
            else if (i == 2 && frames_.find(frame_id) != frames_.end())
                frames_[frame_id]->SetOdom(odom);

            continue;
        }
        if ((i - 3) % 2 == 0)
        {
            input >> temp_data;
            feature_id = std::stoi(temp_data);
            if (features_.find(feature_id) == features_.end())
                features_[feature_id] = std::make_shared<Feature_Id>(feature_id);
            input >> temp_data;
            obs.x() = std::stod(temp_data);
        }
        else
        {
            input >> temp_data;
            input >> temp_data;
            obs.y() = std::stod(temp_data);
            features_[feature_id]->AddFrame(frames_[frame_id]);
            frames_[frame_id]->AddFeature(features_[feature_id], obs);
        }
    }
    f.close();
}
void FeatureFrameManager::GenDataFromTXT(std::string path, bool add_noise)
{
    std::ifstream f;
    f.close();
    f.open(path, std::ios::in);
    std::string line;
    getline(f, line);
    std::cout << line << std::endl;
    int feature_num = std::stoi(line); //特征的个数
    std::cout << feature_num << std::endl;

    getline(f, line);
    std::cout << line << std::endl;
    int frame_num = std::stoi(line); //位姿的个数
    std::cout << frame_num << std::endl;

    if (feature_num == 0 || feature_num == 0)
        return;

    double information_feature, information_post, information_ori; //特征、里程计协方差
    {
        getline(f, line);
        std::stringstream input(line);
        std::string temp_data;
        input >> temp_data;
        std::cout << temp_data << " ";
        information_feature = std::stod(temp_data);
        input >> temp_data;
        std::cout << temp_data << " ";
        information_post = std::stod(temp_data);
        input >> temp_data;
        std::cout << temp_data << std::endl;
        information_ori = std::stod(temp_data);
    }
    static std::default_random_engine e(time(0));
    static std::normal_distribution<double> n_feature(0, information_feature);
    static std::normal_distribution<double> n_post(0, information_post);
    static std::normal_distribution<double> n_ori(0, information_ori);

    //特征x,y的真值
    for (int i = 0; i < feature_num; i++) // line中不包括每行的换行符
    {
        getline(f, line);
        std::stringstream input(line);
        std::string temp_data;
        Eigen::Vector2d fea;
        input >> temp_data;
        // std::cout << temp_data << " ";
        fea.x() = std::stod(temp_data);
        input >> temp_data;
        // std::cout << temp_data << std::endl;
        fea.y() = std::stod(temp_data);
        features_gt_[i] = fea;
        features_[i] = std::make_shared<Feature_Id>(i);
    }
    //
    for (int i = 0; i < frame_num; i++) // line中不包括每行的换行符
    {
        getline(f, line);
        std::stringstream input(line);
        std::string temp_data;
        Eigen::Vector3d frame;
        input >> temp_data;
        // std::cout << temp_data << " ";
        frame.x() = std::stod(temp_data);
        input >> temp_data;
        // std::cout << temp_data << " ";
        frame.y() = std::stod(temp_data);
        input >> temp_data;
        // std::cout << temp_data << std::endl;
        frame.z() = std::stod(temp_data);
        frames_gt_[i] = frame;
        frames_[i] = std::make_shared<Feature_Frame>(i);
    }
    for (int i = 0; i < frame_num; i++) // line中不包括每行的换行符
    {
        Eigen::Vector3d frame = frames_gt_[i];
        Eigen::Matrix2d SO2 = R(frame.z());
        Eigen::Vector2d pos(frame.x(), frame.y());
        getline(f, line);
        std::stringstream input(line);
        std::string temp_data;
        input >> temp_data;
        // std::cout << temp_data << " ";
        int size = std::stoi(temp_data);
        for (int j = 0; j < size; j++)
        {
            input >> temp_data;
            // std::cout << temp_data << std::endl;
            int fea_id = std::stoi(temp_data);
            Eigen::Vector2d mea = SO2.transpose() * (features_gt_[fea_id] - pos);
            // std::cout << mea.transpose() << std::endl;
            if (add_noise)
            {
                Eigen::Vector2d noise = Eigen::Vector2d::Zero().unaryExpr([](double dummy)
                                                                          { return n_feature(e); });
                mea += noise;
            }
            features_[fea_id]->AddFrame(frames_[i]);
            frames_[i]->AddFeature(features_[fea_id], mea);
        }
    }
    for (int i = 0; i < frame_num - 1; i++) // line中不包括每行的换行符
    {
        Eigen::Vector2d pos_i = frames_gt_[i + 1].block(0, 0, 2, 1);
        double ori_i = frames_gt_[i + 1](2);
        Eigen::Vector2d pos_j = frames_gt_[i].block(0, 0, 2, 1);
        double ori_j = frames_gt_[i](2);
        Sophus::SO2d so2_i = Sophus::SO2d::exp(ori_i);
        Sophus::SO2d so2_j = Sophus::SO2d::exp(ori_j);

        Eigen::Vector2d mea_post = so2_j.inverse() * (pos_i - pos_j);
        double mea_oir = (so2_j.inverse() * so2_i).log();
        if (add_noise)
        {
            Eigen::Vector2d noise_post = Eigen::Vector2d::Zero().unaryExpr([](double dummy)
                                                                           { return n_post(e); });
            Eigen::Matrix<double, 1, 1> noise_ori = Eigen::Matrix<double, 1, 1>::Zero().unaryExpr([](double dummy)
                                                                                                  { return n_ori(e); });
            mea_post += noise_post;
            mea_oir += noise_ori(0, 0);
        }
        Eigen::Vector3d mea;
        mea.block(0, 0, 2, 1) = mea_post;
        mea(2) = mea_oir;
        // std::cout << mea.transpose() << std::endl;
        frames_[i]->SetOdom(mea);
    }
}
void FeatureFrameManager::SaveGroundtruth(std::string path)
{
    if (features_gt_.size() == 0 || frames_gt_.size() == 0)
        return;
    std::fstream fs(path, std::ios::out);
    for (auto &fea_gt : features_gt_)
    {
        int index = fea_gt.first;
        auto &fea_xy = fea_gt.second;
        fs << fea_xy.x() << " " << 2 << " " << index << std::endl;
        fs << fea_xy.y() << " " << 2 << " " << index << std::endl;
    }
    for (auto &pos_gt : frames_gt_)
    {
        int index = pos_gt.first;
        if (index != 0)
        {
            auto &pos_xy = pos_gt.second;
            fs << pos_xy.x() << " " << 1 << " " << index << std::endl;
            fs << pos_xy.y() << " " << 1 << " " << index << std::endl;
            fs << pos_xy.z() << " " << 1 << " " << index << std::endl;
        }
    }
}
void FeatureFrameManager::SaveMeasurement(std::string path, int bias)
{
    if (frames_.size() == 0 || features_gt_.size() == 0)
        return;
    std::fstream fs(path, std::ios::app);
    for (auto &frame : frames_)
    {
        int pose_index = frame.first;
        auto fea_map = frame.second->GetFeaturesObs();
        for (auto &fea : fea_map)
        {
            int fea_id = fea.first;
            auto &fea_pos = fea.second;
            fs << fea_pos.x() << " " << 2 << " " << fea_id << " " << pose_index + bias << std::endl;
            fs << fea_pos.y() << " " << 2 << " " << fea_id << " " << pose_index + bias << std::endl;
        }
        if (pose_index < frames_.size()) // - 1
        {
            Eigen::Vector3d odom = frame.second->GetOdom();
            fs << odom.x() << " " << 1 << " " << pose_index + bias + 1 << " " << pose_index + bias << std::endl;
            fs << odom.y() << " " << 1 << " " << pose_index + bias + 1 << " " << pose_index + bias << std::endl;
            fs << odom.z() << " " << 1 << " " << pose_index + bias + 1 << " " << pose_index + bias << std::endl;
        }
    }
}
int FeatureFrameManager::GetFeatureSize()
{
    return features_.size();
}
int FeatureFrameManager::GetFrameSize()
{
    return frames_.size();
}
std::vector<uint> FeatureFrameManager::GetFeaturesId()
{
    std::vector<uint> keys;
    for (auto &feature : features_)
    {
        keys.push_back(feature.first);
    }
    return keys;
}
std::vector<uint> FeatureFrameManager::GetFramesId()
{
    std::vector<uint> keys;
    for (auto &frame : frames_)
    {
        keys.push_back(frame.first);
    }
    return keys;
}
void FeatureFrameManager::MergeFrame()
{
    auto begin_frame = frames_.begin();
    for (auto iter = frames_.begin(); iter != frames_.end(); iter++)
    {
        if (iter == begin_frame)
            continue;
        begin_frame->second->MergeFrame(iter->second);
    }
}
void FeatureFrameManager::ComputeParams(Eigen::MatrixXd &A, Eigen::VectorXd &Z_xy, Eigen::VectorXd &Z_ori, Eigen::MatrixXd &C,
                                        std::map<uint, Eigen::MatrixXd> &Ai, std::map<uint, Eigen::VectorXd> &Zi)
{
    GetA(A, Ai);
    GetZ_xy(Z_xy, Zi);
    GetZ_ori(Z_ori);
    GetC(C);
}
void FeatureFrameManager::ComputeState(Eigen::MatrixXd &A, Eigen::VectorXd &Z, Eigen::MatrixXd &C, std::vector<double> &phis, std::string save_path)
{
    std::vector<uint> fea_id;
    int rows = frames_.size() - 1;
    int cols = frames_.size() - 1;
    for (auto frame : frames_)
    {
        int fea_size = frame.second->GetMulitViewedFeatureNum();
        cols += fea_size;
    }
    for (auto feature : features_)
        if (feature.second->GetFrameNum() > 1)
        {
            fea_id.emplace_back(feature.first);
            rows++;
        }
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(cols * 2, cols * 2);

    int full_obs_size = 0;
    int count = 0;
    for (auto frame : frames_)
    {
        int fea_size = frame.second->GetMulitViewedFeatureNum();
        R.block(full_obs_size * 2, full_obs_size * 2, fea_size * 2, fea_size * 2) = GetR(fea_size, phis[count]);
        full_obs_size += fea_size;
        count++;
    }
    count = 0;
    for (auto frame : frames_)
    {
        if (count >= (int)frames_.size() - 1)
            break;
        R.block(full_obs_size * 2, full_obs_size * 2, 2, 2) = GetR(1, phis[count]);
        count++;
        full_obs_size++;
    }
    full_obs_size -= ((int)frames_.size() - 1);

    Eigen::SparseMatrix<double> A_spa = A.sparseView();
    Eigen::SparseMatrix<double> C_spa = C.sparseView();
    Eigen::SparseMatrix<double> R_spa = R.sparseView();
    Eigen::SparseVector<double> Z_spa = Z.sparseView();

    Eigen::MatrixXd ACAT = A_spa * C_spa * A_spa.transpose();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(ACAT.rows(), ACAT.cols());
    Eigen::SparseMatrix<double> acat = ACAT.sparseView();
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::AMDOrdering<int>> QRsolver;
    QRsolver.compute(acat);
    Eigen::SparseMatrix<double> ACAT_inv = QRsolver.solve(I.sparseView());
    Eigen::VectorXd X = ACAT_inv * A_spa * C_spa * R_spa * Z_spa;

    std::map<uint, Eigen::Vector2d> fea_post;

    for (count = 0; count < (int)fea_id.size(); count++)
    {
        fea_post[fea_id[count]] = X.block(count * 2, 0, 2, 1);
    }
    //将所有只有本帧观测到的特征转换到世界坐标系
    count = 0;
    for (auto frame : frames_)
    {
        std::vector<uint> single_ids = frame.second->GetSingleViewedFeaturesId();
        Eigen::Matrix2d R_wr = GetR(1, phis[count]);
        if (count > 0)
            std::cout << X.block((fea_id.size() + count - 1) * 2, 0, 2, 1).transpose() << std::endl;
        for (uint id : single_ids)
        {
            Eigen::Vector2d obs = frame.second->GetFeatureObs(id);

            if (count > 0)
            {
                std::cout << id << " ";
                fea_post[id] = R_wr * obs;
                fea_post[id] += X.block((fea_id.size() + count - 1) * 2, 0, 2, 1); // X_w = R_wr*X_r + t_wr
            }
            else
            {
                fea_post[id] = obs;
            }
        }
        std::cout << std::endl;
        count++;
    }

    Eigen::Vector3d odom;
    Eigen::Vector2d X_wrn1 = X.tail(2);
    Eigen::Vector2d t_rn1rn2 = (frames_.rbegin()->second->GetOdom()).block(0, 0, 2, 1);
    Eigen::Matrix2d R_wrn1 = GetR(1, phis.back());

    Eigen::Vector2d X_wrn2 = R_wrn1 * t_rn1rn2 + X_wrn1;
    double phi_rn2 = phis.back() + frames_.rbegin()->second->GetOdom()(2);
    // std::cout << "phi_rn2:" << phi_rn2 << std::endl;
    if (phi_rn2 > M_PI)
    {
        phi_rn2 -= 2 * M_PI;
    }
    else if (phi_rn2 < -M_PI)
    {
        phi_rn2 += 2 * M_PI;
    }
    odom.block(0, 0, 2, 1) = X_wrn2;
    odom(2) = phi_rn2;

    odom.block(0, 0, 2, 1) = X_wrn1;
    odom(2) = phis.back();

    std::ofstream f;
    f.close();
    f.open(save_path, std::ios::out);
    f << 0 << " " << std::setprecision(10) << odom(0) << "\n";
    f << 0 << " " << std::setprecision(10) << odom(1) << "\n";
    f << 0 << " " << std::setprecision(10) << odom(2) << "\n";
    for (auto &fea : fea_post)
    {
        f << fea.first << " " << std::setprecision(10) << fea.second(0) << "\n";
        f << fea.first << " " << std::setprecision(10) << fea.second(1) << "\n";
    }
}
void FeatureFrameManager::ComputeState(Eigen::MatrixXd &A, Eigen::VectorXd &Z, Eigen::MatrixXd &C, std::vector<double> &phis, Eigen::VectorXd &X_)
{
    std::vector<uint> fea_id;
    int rows = frames_.size() - 1;
    int cols = frames_.size() - 1;
    for (auto frame : frames_)
    {
        int fea_size = frame.second->GetMulitViewedFeatureNum();
        cols += fea_size;
    }
    for (auto feature : features_)
        if (feature.second->GetFrameNum() > 1)
        {
            fea_id.emplace_back(feature.first);
            rows++;
        }
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(cols * 2, cols * 2);

    int full_obs_size = 0;
    int count = 0;
    for (auto frame : frames_)
    {
        int fea_size = frame.second->GetMulitViewedFeatureNum();
        R.block(full_obs_size * 2, full_obs_size * 2, fea_size * 2, fea_size * 2) = GetR(fea_size, phis[count]);
        full_obs_size += fea_size;
        count++;
    }
    count = 0;
    for (auto frame : frames_)
    {
        if (count >= (int)frames_.size() - 1)
            break;
        R.block(full_obs_size * 2, full_obs_size * 2, 2, 2) = GetR(1, phis[count]);
        // std::cout << GetR(1, phis[count]) << std::endl;
        count++;
        full_obs_size++;
    }
    full_obs_size -= ((int)frames_.size() - 1);

    Eigen::SparseMatrix<double> A_spa = A.sparseView();
    Eigen::SparseMatrix<double> C_spa = C.sparseView();
    Eigen::SparseMatrix<double> R_spa = R.sparseView();
    Eigen::SparseVector<double> Z_spa = Z.sparseView();
    Eigen::MatrixXd ACAT = A_spa * C_spa * A_spa.transpose();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(ACAT.rows(), ACAT.cols());
    Eigen::SparseMatrix<double> acat = ACAT.sparseView();
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::AMDOrdering<int>> QRsolver;
    QRsolver.compute(acat);
    Eigen::SparseMatrix<double> ACAT_inv = QRsolver.solve(I.sparseView());
    Eigen::VectorXd X = ACAT_inv * A_spa * C_spa * R_spa * Z_spa;
    X_ = X;
}
void FeatureFrameManager::GetA(Eigen::MatrixXd &A_, std::map<uint, Eigen::MatrixXd> &Ai_)
{
    Eigen::MatrixXd A;
    std::map<uint, Eigen::MatrixXd> Ai;
    int rows = frames_.size() - 1;
    int cols = frames_.size() - 1;
    int full_obs_size = 0;
    for (auto frame : frames_)
    {
        int fea_size = frame.second->GetMulitViewedFeatureNum();
        cols += fea_size;
        full_obs_size += fea_size;
    }
    for (auto feature : features_)
        if (feature.second->GetFrameNum() > 1)
            rows++;

    A = Eigen::MatrixXd::Zero(rows, cols);
    int row = 0;
    int col = 0;
    for (auto feature : features_)
    {
        if (feature.second->GetFrameNum() <= 1)
            continue;

        uint feature_id = feature.second->GetFeatureId();
        for (auto frame : frames_)
        {
            if (frame.second->HasFeatureId(feature_id))
            {
                auto feas = frame.second->GetMulitViewedFeaturesId();
                for (size_t i = 0; i < feas.size(); i++)
                {
                    if (feas[i] == feature_id)
                    {
                        A(row, col + i) = 1;
                        break;
                    }
                }
            }
            col += frame.second->GetMulitViewedFeatureNum();
        }
        col = 0;
        row++;
    }
    uint first_id = frames_.begin()->first;
    int bias = 0;
    for (auto frame : frames_)
    {
        if (frame.first == first_id)
        {
            col += frame.second->GetMulitViewedFeatureNum();
            continue;
        }
        int feas = frame.second->GetMulitViewedFeatureNum();
        for (int i = 0; i < feas; i++)
            A(row, col + i) = -1;
        col += feas;
        A(row, full_obs_size + bias) = 1;
        if (full_obs_size + bias + 1 < cols)
            A(row, full_obs_size + bias + 1) = -1;
        bias++;
        row++;
    }
    A_ = Eigen::kroneckerProduct(A, Eigen::Matrix2d::Identity());
    col = 0;
    for (auto frame : frames_)
    {
        uint frame_id = frame.first;
        int col_num = frame.second->GetMulitViewedFeatureNum();
        Eigen::MatrixXd A_i = Eigen::MatrixXd::Zero(rows, cols);
        A_i.block(0, col, rows, col_num) = A.block(0, col, rows, col_num);
        Ai[frame_id] = A_i;
        col += col_num;
    }
    for (auto frame : frames_)
    {
        uint frame_id = frame.first;
        if (col < cols)
            Ai[frame_id].block(0, col, rows, 1) = A.block(0, col, rows, 1);
        Ai_[frame_id] = Eigen::kroneckerProduct(Ai[frame_id], Eigen::Matrix2d::Identity());
        col++;
    }
}
void FeatureFrameManager::GetZ_xy(Eigen::VectorXd &Z_xy, std::map<uint, Eigen::VectorXd> &Zi)
{
    int rows = frames_.size() - 1;
    for (auto frame : frames_)
    {
        int fea_size = frame.second->GetMulitViewedFeatureNum();
        rows += fea_size;
    }
    Z_xy = Eigen::VectorXd::Zero(rows * 2);
    int row = 0;
    for (auto frame : frames_)
    {
        auto features_id = frame.second->GetMulitViewedFeaturesId();
        for (auto id : features_id)
        {
            auto obs = frame.second->GetFeatureObs(id);
            Z_xy.block(row, 0, 2, 1) = obs;
            row += 2;
        }
    }
    for (auto frame : frames_)
    {
        if (row >= rows * 2)
            break;
        Z_xy.block(row, 0, 2, 1) = frame.second->GetOdom().block(0, 0, 2, 1);
        row += 2;
    }
    row = 0;
    for (auto frame : frames_)
    {
        uint frame_id = frame.first;
        int row_num = frame.second->GetMulitViewedFeatureNum();
        Eigen::VectorXd Z_i = Eigen::VectorXd::Zero(rows * 2);
        Z_i.block(row * 2, 0, row_num * 2, 1) = Z_xy.block(row * 2, 0, row_num * 2, 1);
        Zi[frame_id] = Z_i;
        row += row_num;
    }
    for (auto frame : frames_)
    {
        if (row >= rows)
            break;
        uint frame_id = frame.first;
        Zi[frame_id].block(row * 2, 0, 2, 1) = Z_xy.block(row * 2, 0, 2, 1);
        row++;
    }
}
void FeatureFrameManager::GetZ_ori(Eigen::VectorXd &Z_ori)
{
    Z_ori = Eigen::VectorXd::Zero(frames_.size() - 1);
    int count = 0;
    for (auto frame : frames_)
    {
        if (count >= (int)(frames_.size()) - 1)
            break;
        Z_ori(count) = frame.second->GetOdom()(2);
        count++;
    }
}
void FeatureFrameManager::GetC(Eigen::MatrixXd &C)
{
    int rows = frames_.size() - 1;
    for (auto frame : frames_)
    {
        int fea_size = frame.second->GetMulitViewedFeatureNum();
        rows += fea_size;
    }
    C = Eigen::MatrixXd::Identity(rows * 2, rows * 2);
}
Eigen::MatrixXd FeatureFrameManager::GetR(int size, double phi)
{
    Eigen::Matrix2d R;
    R << cos(phi), -sin(phi), sin(phi), cos(phi);
    Eigen::MatrixXd R_hat = Eigen::MatrixXd::Zero(size * 2, size * 2);
    for (int i = 0; i < size; i++)
    {
        R_hat.block(i * 2, i * 2, 2, 2) = R;
    }
    return R_hat;
}

Feature_Frame::Feature_Frame(int frame_id) : frame_id_(frame_id)
{
    R_ = Eigen::Matrix2d::Identity();
    t_ = Eigen::Vector2d::Zero();
    odom_ = Eigen::Vector3d::Zero();
}
Feature_Frame::Feature_Frame(int frame_id, Eigen::Vector3d odom) : frame_id_(frame_id), odom_(odom)
{
    R_ = Eigen::Matrix2d::Identity();
    t_ = Eigen::Vector2d::Zero();
}
void Feature_Frame::AddFeature(std::shared_ptr<Feature_Id> feature, Eigen::Vector2d obs)
{
    obs_f_[feature->GetFeatureId()] = obs;
    features_[feature->GetFeatureId()] = feature;
}
void Feature_Frame::SetOdom(Eigen::Vector3d odom)
{
    odom_ = odom;
}
void Feature_Frame::SetPose(Eigen::Matrix2d R, Eigen::Vector2d t)
{
    R_ = R;
    t_ = t;
}
void Feature_Frame::reset()
{
    std::map<uint, Eigen::Vector2d>().swap(obs_f_);
    std::map<uint, std::weak_ptr<Feature_Id>>().swap(features_);
    malloc_trim(0);
    R_ = Eigen::Matrix2d::Identity();
    t_ = Eigen::Vector2d::Zero();
    odom_ = Eigen::Vector3d::Zero();
}
void Feature_Frame::MergeFrame(std::shared_ptr<Feature_Frame> other)
{
    auto other_features = other->GetFeatures();
    for (auto other_feature : other_features)
    {
        if (std::shared_ptr<Feature_Id> s_other_feature = other_feature.lock())
        {
            uint f_id = s_other_feature->GetFeatureId();
            if (features_.find(f_id) != features_.end())
                continue;

            features_[f_id] = other_feature;
            obs_f_[f_id] = s_other_feature->GetFeatureWorldPosition();
        }
    }
    other->reset();
}
uint Feature_Frame::GetFrameId()
{
    return frame_id_;
}
int Feature_Frame::GetMulitViewedFeatureNum()
{
    int count = 0;
    for (auto feature : features_)
    {
        if (std::shared_ptr<Feature_Id> s_feature = feature.second.lock())
            if (s_feature->GetFrameNum() > 1)
                count++;
    }
    return count;
}
int Feature_Frame::GetSingleViewedFeatureNum()
{
    int count = 0;
    for (auto feature : features_)
    {
        if (std::shared_ptr<Feature_Id> s_feature = feature.second.lock())
            if (s_feature->GetFrameNum() == 1)
                count++;
    }
    return count;
}
int Feature_Frame::GetFeatureNum()
{
    return features_.size();
}
bool Feature_Frame::HasFeatureId(uint id)
{
    return features_.find(id) != features_.end();
}
std::vector<std::weak_ptr<Feature_Id>> Feature_Frame::GetFeatures()
{
    std::vector<std::weak_ptr<Feature_Id>> features;
    for (auto feature : features_)
    {
        features.push_back(feature.second);
    }
    return features;
}
std::vector<std::weak_ptr<Feature_Id>> Feature_Frame::GetMulitViewedFeatures()
{
    std::vector<std::weak_ptr<Feature_Id>> features;
    for (auto feature : features_)
        if (std::shared_ptr<Feature_Id> s_feature = feature.second.lock())
            if (s_feature->GetFrameNum() > 1)
                features.push_back(feature.second);

    return features;
}
std::vector<std::weak_ptr<Feature_Id>> Feature_Frame::GetSingleViewedFeatures()
{
    std::vector<std::weak_ptr<Feature_Id>> features;
    for (auto feature : features_)
        if (std::shared_ptr<Feature_Id> s_feature = feature.second.lock())
            if (s_feature->GetFrameNum() == 1)
                features.push_back(feature.second);

    return features;
}
std::vector<uint> Feature_Frame::GetFeaturesId()
{
    std::vector<uint> keys;
    for (auto &feature : obs_f_)
        keys.push_back(feature.first);

    return keys;
}
std::vector<uint> Feature_Frame::GetMulitViewedFeaturesId()
{
    std::vector<uint> keys;
    for (auto feature : features_)
        if (std::shared_ptr<Feature_Id> s_feature = feature.second.lock())
            if (s_feature->GetFrameNum() > 1)
                keys.push_back(feature.first);

    return keys;
}
std::vector<uint> Feature_Frame::GetSingleViewedFeaturesId()
{
    std::vector<uint> keys;
    for (auto feature : features_)
        if (std::shared_ptr<Feature_Id> s_feature = feature.second.lock())
            if (s_feature->GetFrameNum() == 1)
                keys.push_back(feature.first);

    return keys;
}
std::map<uint, Eigen::Vector2d> Feature_Frame::GetFeaturesObs()
{
    return obs_f_;
}
Eigen::Vector3d Feature_Frame::GetOdom()
{
    return odom_;
}
Eigen::Vector2d Feature_Frame::GetFeatureObs(uint feature_id)
{
    if (obs_f_.find(feature_id) != obs_f_.end())
        return obs_f_[feature_id];
    else
        return Eigen::Vector2d();
}

Feature_Id::Feature_Id(int feature_id) : feature_id_(feature_id)
{
    xy_w_ = Eigen::Vector2d::Zero();
}
void Feature_Id::AddFrame(std::shared_ptr<Feature_Frame> frame)
{
    frames_[frame->GetFrameId()] = frame;
}
void Feature_Id::reset()
{
    std::map<uint, std::weak_ptr<Feature_Frame>>().swap(frames_);
    malloc_trim(0);
}
void Feature_Id::SetWorldPostion(Eigen::Vector2d xy)
{
    xy_w_ = xy;
}
uint Feature_Id::GetFeatureId()
{
    return feature_id_;
}
int Feature_Id::GetFrameNum()
{
    return frames_.size();
}
Eigen::Vector2d Feature_Id::GetFeatureWorldPosition()
{
    return xy_w_;
}
std::vector<std::weak_ptr<Feature_Frame>> Feature_Id::GetFrames()
{
    std::vector<std::weak_ptr<Feature_Frame>> frames;
    for (auto frame : frames_)
    {
        frames.push_back(frame.second);
    }
    return frames;
}
std::vector<uint> Feature_Id::GetFramesId()
{
    std::vector<uint> keys;
    for (auto &frame : frames_)
    {
        keys.push_back(frame.first);
    }
    return keys;
}