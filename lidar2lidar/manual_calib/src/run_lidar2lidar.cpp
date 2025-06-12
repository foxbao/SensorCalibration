/*
 * Copyright (C) 2021 by Autonomous Driving Group, Shanghai AI Laboratory
 * Limited. All rights reserved.
 * Yan Guohang <yanguohang@pjlab.org.cn>
 * Ouyang Jinhua <ouyangjinhua@pjlab.org.cn>
 */
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>
#include <pcl/common/transforms.h>
#include <pcl/conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <Eigen/Core>
#include <iostream>
#include <string>
#include <Eigen/Geometry>
#include "extrinsic_param.hpp"
#include <nlohmann/json.hpp>  // JSON 头文件：https://github.com/nlohmann/json
using json = nlohmann::json;
using namespace std;

#define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049
#define MAX_RADAR_TIME_GAP 15 * 1e6
// #define APPLY_COLOR_TO_LIDAR_INTENSITY  // to set intensity colored or not

pangolin::GlBuffer *source_vertexBuffer_;
pangolin::GlBuffer *source_colorBuffer_;
pangolin::GlBuffer *target_vertexBuffer_;
pangolin::GlBuffer *target_colorBuffer_;

double cali_scale_degree_ = 0.1;
double cali_scale_trans_ = 0.03;
static Eigen::Matrix4d calibration_matrix_ = Eigen::Matrix4d::Identity();
static Eigen::Matrix4d orign_calibration_matrix_ = Eigen::Matrix4d::Identity();
// std::vector<Eigen::Matrix4d> modification_list_;
std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> modification_list_;
bool display_mode_ = false;
int point_size_ = 2;

struct RGB {
  unsigned char r;
  unsigned char g;
  unsigned char b;
};

bool kbhit() {
  termios term;
  tcgetattr(0, &term);
  termios term2 = term;
  term2.c_lflag &= ~ICANON;
  tcsetattr(0, TCSANOW, &term2);
  int byteswaiting;
  ioctl(0, FIONREAD, &byteswaiting);
  tcsetattr(0, TCSANOW, &term);
  return byteswaiting > 0;
}

void CalibrationInit(Eigen::Matrix4d json_param) {
  Eigen::Matrix4d init_cali;
  init_cali << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
  calibration_matrix_ = json_param;
  orign_calibration_matrix_ = json_param;
  modification_list_.reserve(12);
  for (int32_t i = 0; i < 12; i++) {
    std::vector<int> transform_flag(6, 0);
    transform_flag[i / 2] = (i % 2) ? (-1) : 1;
    Eigen::Matrix4d tmp = Eigen::Matrix4d::Identity();
    Eigen::Matrix3d rot_tmp;
    rot_tmp =
        Eigen::AngleAxisd(transform_flag[0] * cali_scale_degree_ / 180.0 * M_PI,
                          Eigen::Vector3d::UnitX()) *
        Eigen::AngleAxisd(transform_flag[1] * cali_scale_degree_ / 180.0 * M_PI,
                          Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(transform_flag[2] * cali_scale_degree_ / 180.0 * M_PI,
                          Eigen::Vector3d::UnitZ());
    tmp.block(0, 0, 3, 3) = rot_tmp;
    tmp(0, 3) = transform_flag[3] * cali_scale_trans_;
    tmp(1, 3) = transform_flag[4] * cali_scale_trans_;
    tmp(2, 3) = transform_flag[5] * cali_scale_trans_;
    // std::cout<<modification_list_[0]<<std::endl;
    std::cout<<tmp<<std::endl;
    modification_list_[i] = tmp;
  }
  std::cout << "=>Calibration scale Init!\n";
}

void DrawCoordinateAxis(float length = 5.0f)
{
    glLineWidth(3.0f);
    glBegin(GL_LINES);

    // X axis - Red
    glColor3f(1.0f, 0.0f, 0.0f);  // 红色
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(length, 0.0f, 0.0f);

    // Y axis - Green
    glColor3f(0.0f, 1.0f, 0.0f);  // 绿色
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, length, 0.0f);

    // Z axis - Blue
    glColor3f(0.0f, 0.0f, 1.0f);  // 蓝色
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, length);

    glEnd();
}

void DrawTransformedAxis(const Eigen::Matrix4d& transform, float length = 1.0f)
{
    glLineWidth(3.0f);
    glBegin(GL_LINES);

    // 原点
    Eigen::Vector4d origin(0, 0, 0, 1);
    Eigen::Vector4d x_end(length, 0, 0, 1);
    Eigen::Vector4d y_end(0, length, 0, 1);
    Eigen::Vector4d z_end(0, 0, length, 1);

    // 变换后的位置
    Eigen::Vector4d o_t = transform * origin;
    Eigen::Vector4d x_t = transform * x_end;
    Eigen::Vector4d y_t = transform * y_end;
    Eigen::Vector4d z_t = transform * z_end;

    // X轴 - 红
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(o_t.x(), o_t.y(), o_t.z());
    glVertex3f(x_t.x(), x_t.y(), x_t.z());

    // Y轴 - 绿
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(o_t.x(), o_t.y(), o_t.z());
    glVertex3f(y_t.x(), y_t.y(), y_t.z());

    // Z轴 - 蓝
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(o_t.x(), o_t.y(), o_t.z());
    glVertex3f(z_t.x(), z_t.y(), z_t.z());

    glEnd();
}

void CalibrationScaleChange() {
  Eigen::Matrix4d init_cali;
  init_cali << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
  modification_list_.reserve(12);
  for (int32_t i = 0; i < 12; i++) {
    std::vector<int> transform_flag(6, 0);
    transform_flag[i / 2] = (i % 2) ? (-1) : 1;
    Eigen::Matrix4d tmp = Eigen::Matrix4d::Identity();
    Eigen::Matrix3d rot_tmp;
    rot_tmp =
        Eigen::AngleAxisd(transform_flag[0] * cali_scale_degree_ / 180.0 * M_PI,
                          Eigen::Vector3d::UnitX()) *
        Eigen::AngleAxisd(transform_flag[1] * cali_scale_degree_ / 180.0 * M_PI,
                          Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(transform_flag[2] * cali_scale_degree_ / 180.0 * M_PI,
                          Eigen::Vector3d::UnitZ());
    tmp.block(0, 0, 3, 3) = rot_tmp;
    tmp(0, 3) = transform_flag[3] * cali_scale_trans_;
    tmp(1, 3) = transform_flag[4] * cali_scale_trans_;
    tmp(2, 3) = transform_flag[5] * cali_scale_trans_;
    modification_list_[i] = tmp;
  }
  std::cout << "=>Calibration scale update done!\n";
}

std::string GetFileBaseName(const std::string& path) {
    // 取文件名部分
    auto pos_slash = path.find_last_of("/\\");
    std::string filename = (pos_slash == std::string::npos) ? path : path.substr(pos_slash + 1);
    
    // 去掉扩展名
    auto pos_dot = filename.find_last_of(".");
    return (pos_dot == std::string::npos) ? filename : filename.substr(0, pos_dot);
}

void saveResult(const int &frame_id, const std::string &base_filename = "")
{
    // 1. 构造文件名
    std::string txt_file = base_filename.empty()
                               ? "lidar2lidar_extrinsic_" + std::to_string(frame_id) + ".txt"
                               : base_filename + ".txt";
    std::string json_file = base_filename.empty()
                                ? "lidar2lidar_extrinsic_" + std::to_string(frame_id) + ".json"
                                : base_filename + ".json";

    // 2. 打开 .txt 文件
    std::ofstream fCalib(txt_file);
    if (!fCalib.is_open())
    {
        std::cerr << "open file " << txt_file << " failed." << std::endl;
        return;
    }

    // 3. 提取数据
    Eigen::Matrix3d R = calibration_matrix_.block<3, 3>(0, 0);
    Eigen::Vector3d t = calibration_matrix_.block<3, 1>(0, 3);
    Eigen::Quaterniond q(R);
    Eigen::Vector3d euler_angles = R.eulerAngles(2, 1, 0); // yaw(Z), pitch(Y), roll(X)
    double roll_deg = euler_angles(2) * 180.0 / M_PI;
    double pitch_deg = euler_angles(1) * 180.0 / M_PI;
    double yaw_deg = euler_angles(0) * 180.0 / M_PI;

    // 4. 写入 txt 文件
    fCalib << "Extrinsic:" << std::endl;
    fCalib << "R:\n"
           << R(0, 0) << " " << R(0, 1) << " " << R(0, 2) << "\n"
           << R(1, 0) << " " << R(1, 1) << " " << R(1, 2) << "\n"
           << R(2, 0) << " " << R(2, 1) << " " << R(2, 2) << std::endl;
    fCalib << "t: " << t.transpose() << std::endl;

    fCalib << "RPY (degrees):" << std::endl;
    fCalib << "Roll (X): " << roll_deg << std::endl;
    fCalib << "Pitch (Y): " << pitch_deg << std::endl;
    fCalib << "Yaw (Z): " << yaw_deg << std::endl;

    fCalib << "Quaternion (x, y, z, w):" << std::endl;
    fCalib << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;

    fCalib << "************* json format *************" << std::endl;
    fCalib << "Extrinsic (t + q):" << std::endl;
    fCalib << "[" << t.x() << ", " << t.y() << ", " << t.z() << ", "
           << q.x() << ", " << q.y() << ", " << q.z() << ", " << q.w() << "]" << std::endl;
    fCalib << "Extrinsic Matrix:" << std::endl;
    fCalib << "["
           << calibration_matrix_(0,0) << "," << calibration_matrix_(0,1) << "," << calibration_matrix_(0,2) << "," << calibration_matrix_(0,3) << "],"
           << "["
           << calibration_matrix_(1,0) << "," << calibration_matrix_(1,1) << "," << calibration_matrix_(1,2) << "," << calibration_matrix_(1,3) << "],"
           << "["
           << calibration_matrix_(2,0) << "," << calibration_matrix_(2,1) << "," << calibration_matrix_(2,2) << "," << calibration_matrix_(2,3) << "],"
           << "["
           << calibration_matrix_(3,0) << "," << calibration_matrix_(3,1) << "," << calibration_matrix_(3,2) << "," << calibration_matrix_(3,3) << "]"
           << std::endl;
    fCalib.close();

    // 5. 写入 JSON 文件
    json j;
    j["comment"] = "transform = [tx, ty, tz, qx, qy, qz, qw]";
    j["transform"] = {
        t.x(), t.y(), t.z(),
        q.x(), q.y(), q.z(), q.w()
    };

    std::ofstream json_out(json_file);
    if (!json_out.is_open())
    {
        std::cerr << "open file " << json_file << " failed." << std::endl;
        return;
    }
    json_out << j.dump(4);
    json_out.close();

    // 6. 控制台输出
    std::cout << "✅ 保存外参成功：" << txt_file << " 和 " << json_file << std::endl;
    std::cout << "RPY (deg): Roll=" << roll_deg << ", Pitch=" << pitch_deg << ", Yaw=" << yaw_deg << std::endl;
    std::cout << "Quaternion (x, y, z, w): " << q.x() << ", " << q.y() << ", " << q.z() << ", " << q.w() << std::endl;
}

bool ManualCalibration(int key_input) {
  char table[] = {'q', 'a', 'w', 's', 'e', 'd', 'r', 'f', 't', 'g', 'y', 'h'};
  bool real_hit = false;
  for (int32_t i = 0; i < 12; i++) {
    if (key_input == table[i]) {
      calibration_matrix_ = calibration_matrix_ * modification_list_[i];
      real_hit = true;
    }
  }
  return real_hit;
}

RGB GreyToColorMix(int val) {
  int r, g, b;
  if (val < 128) {
    r = 0;
  } else if (val < 192) {
    r = 255 / 64 * (val - 128);
  } else {
    r = 255;
  }
  if (val < 64) {
    g = 255 / 64 * val;
  } else if (val < 192) {
    g = 255;
  } else {
    g = -255 / 63 * (val - 192) + 255;
  }
  if (val < 64) {
    b = 255;
  } else if (val < 128) {
    b = -255 / 63 * (val - 192) + 255;
  } else {
    b = 0;
  }
  RGB rgb;
  rgb.b = b;
  rgb.g = g;
  rgb.r = r;
  return rgb;
}

void ProcessTargetFrame(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloudLidar,
                        const bool &diaplay_mode) {
  if (target_vertexBuffer_ != nullptr)
    delete (target_vertexBuffer_);
  if (target_colorBuffer_ != nullptr)
    delete (target_colorBuffer_);
  int pointsNum = cloudLidar->points.size();
  pangolin::GlBuffer *vertexbuffer = new pangolin::GlBuffer(
      pangolin::GlArrayBuffer, pointsNum, GL_FLOAT, 3, GL_DYNAMIC_DRAW);
  pangolin::GlBuffer *colorbuffer = new pangolin::GlBuffer(
      pangolin::GlArrayBuffer, pointsNum, GL_UNSIGNED_BYTE, 3, GL_DYNAMIC_DRAW);

  float *dataUpdate = new float[pointsNum * 3];
  unsigned char *colorUpdate = new unsigned char[pointsNum * 3];
  for (int ipt = 0; ipt < pointsNum; ipt++) {
    Eigen::Vector4d pointPos(cloudLidar->points[ipt].x,
                             cloudLidar->points[ipt].y,
                             cloudLidar->points[ipt].z, 1.0);
    dataUpdate[ipt * 3 + 0] = pointPos.x();
    dataUpdate[ipt * 3 + 1] = pointPos.y();
    dataUpdate[ipt * 3 + 2] = pointPos.z();

    if (diaplay_mode) {
      RGB colorFake = GreyToColorMix(cloudLidar->points[ipt].intensity);
      colorUpdate[ipt * 3 + 0] = static_cast<unsigned char>(colorFake.r);
      colorUpdate[ipt * 3 + 1] = static_cast<unsigned char>(colorFake.g);
      colorUpdate[ipt * 3 + 2] = static_cast<unsigned char>(colorFake.b);
    } else {
      // for (int k = 0; k < 3; k++) {
      //   colorUpdate[ipt * 3 + k] =
      //       static_cast<unsigned char>(cloudLidar->points[ipt].intensity);
      // }
      colorUpdate[ipt * 3 + 0] = 0;    // R
      colorUpdate[ipt * 3 + 1] = 255;  // G
      colorUpdate[ipt * 3 + 2] = 255;  // B
    }
  }

  (vertexbuffer)->Upload(dataUpdate, sizeof(float) * 3 * pointsNum, 0);
  (colorbuffer)->Upload(colorUpdate, sizeof(unsigned char) * 3 * pointsNum, 0);

  target_vertexBuffer_ = vertexbuffer;
  target_colorBuffer_ = colorbuffer;
  std::cout << "Process target lidar frame!\n";
}

void ProcessSourceFrame(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloudLidar,
                        const Eigen::Matrix4d &extrinsic,
                        const bool &diaplay_mode) {
  if (source_vertexBuffer_ != nullptr)
    delete (source_vertexBuffer_);
  if (source_colorBuffer_ != nullptr)
    delete (source_colorBuffer_);
  int pointsNum = cloudLidar->points.size();
  pangolin::GlBuffer *vertexbuffer = new pangolin::GlBuffer(
      pangolin::GlArrayBuffer, pointsNum, GL_FLOAT, 3, GL_DYNAMIC_DRAW);
  pangolin::GlBuffer *colorbuffer = new pangolin::GlBuffer(
      pangolin::GlArrayBuffer, pointsNum, GL_UNSIGNED_BYTE, 3, GL_DYNAMIC_DRAW);

  float *dataUpdate = new float[pointsNum * 3];
  unsigned char *colorUpdate = new unsigned char[pointsNum * 3];
  for (int ipt = 0; ipt < pointsNum; ipt++) {
    Eigen::Vector4d pointPos(cloudLidar->points[ipt].x,
                             cloudLidar->points[ipt].y,
                             cloudLidar->points[ipt].z, 1.0);
    Eigen::Vector4d trans_p = extrinsic * pointPos;
    dataUpdate[ipt * 3 + 0] = trans_p.x();
    dataUpdate[ipt * 3 + 1] = trans_p.y();
    dataUpdate[ipt * 3 + 2] = trans_p.z();

    if (diaplay_mode) {
      RGB colorFake = GreyToColorMix(cloudLidar->points[ipt].intensity);
      colorUpdate[ipt * 3 + 0] = static_cast<unsigned char>(colorFake.r);
      colorUpdate[ipt * 3 + 1] = static_cast<unsigned char>(colorFake.b);
      colorUpdate[ipt * 3 + 2] = static_cast<unsigned char>(colorFake.g);
    } else {
      // red
      // colorUpdate[ipt * 3 + 0] =
      //     static_cast<unsigned char>(cloudLidar->points[ipt].intensity);
      // colorUpdate[ipt * 3 + 1] = 0;
      // colorUpdate[ipt * 3 + 2] = 0;
      colorUpdate[ipt * 3 + 0] = 255;  // R
      colorUpdate[ipt * 3 + 1] = 0;    // G
      colorUpdate[ipt * 3 + 2] = 0;    // B
    }
  }

  (vertexbuffer)->Upload(dataUpdate, sizeof(float) * 3 * pointsNum, 0);
  (colorbuffer)->Upload(colorUpdate, sizeof(unsigned char) * 3 * pointsNum, 0);

  source_vertexBuffer_ = vertexbuffer;
  source_colorBuffer_ = colorbuffer;
  std::cout << "Process target lidar frame!\n";
}



int main(int argc, char **argv) {

  if (argc != 4) {
    cout << "Usage: ./run_lidar2lidar <target_pcd_path> <source_pcd_path> <extrinsic_json>"
              "\nexample:\n\t"
              "./bin/run_lidar2lidar data/qt.pcd data/p64.pcd data/p64-to-qt-extrinsic.json"
            << endl;
    return 0;
  }

  std::cout <<EIGEN_WORLD_VERSION;
    // std::cout << "Eigen version: " << EIGEN_MAJOR_VERSION << "." 
    //           << EIGEN_MINOR_VERSION << "." 
    //           << std::endl;

  string target_lidar_path = argv[1];
  string source_lidar_path = argv[2];
  string extrinsic_json = argv[3];
  // load target lidar points
  pcl::PointCloud<pcl::PointXYZI>::Ptr target_cloud(
      new pcl::PointCloud<pcl::PointXYZI>);
  if (pcl::io::loadPCDFile<pcl::PointXYZI>(target_lidar_path, *target_cloud) ==
      -1) {
    PCL_ERROR("Couldn't read target lidar file \n");
    return (-1);
  }
  // load source lidar points
  pcl::PointCloud<pcl::PointXYZI>::Ptr source_cloud(
      new pcl::PointCloud<pcl::PointXYZI>);
  if (pcl::io::loadPCDFile<pcl::PointXYZI>(source_lidar_path, *source_cloud) ==
      -1) {
    PCL_ERROR("Couldn't read source lidar file \n");
    return (-1);
  }
  // load extrinsic
  Eigen::Matrix4d json_param;
  LoadExtrinsic(extrinsic_json, json_param);
  std::cout << "lidar to lidar extrinsic:\n" << json_param << std::endl;

  cout << "Loading data completed!" << endl;
  CalibrationInit(json_param);
  const int width = 1920, height = 1280;
  pangolin::CreateWindowAndBind("lidar2lidar player", width, height);

  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_TRUE);
  glDepthFunc(GL_LESS);

  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
      // pangolin::ModelViewLookAt(0, 0, 100, 0, 0, 0, 0.0, 1.0, 0.0));
      pangolin::ModelViewLookAt(0, 100, 0, 0, 0, 0, 0.0, 0.0, 1.0));
  pangolin::View &d_cam = pangolin::CreateDisplay()
                              .SetBounds(0.0, 1.0, pangolin::Attach::Pix(150),
                                         1.0, -1.0 * width / height)
                              .SetHandler(new pangolin::Handler3D(s_cam));
  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  pangolin::OpenGlMatrix Twc; // camera to world
  Twc.SetIdentity();

  // control panel
  pangolin::CreatePanel("cp").SetBounds(pangolin::Attach::Pix(30), 1.0, 0.0,
                                        pangolin::Attach::Pix(150));
  pangolin::Var<bool> displayMode("cp.Intensity Color", display_mode_,
                                  true); // logscale
  pangolin::Var<int> pointSize("cp.Point Size", 2, 0, 8);
  pangolin::Var<double> degreeStep("cp.deg step", cali_scale_degree_, 0,
                                   1); // logscale
  pangolin::Var<double> tStep("cp.t step(cm)", 2, 0, 15);

  pangolin::Var<bool> addXdegree("cp.+ x degree", false, false);
  pangolin::Var<bool> minusXdegree("cp.- x degree", false, false);
  pangolin::Var<bool> addYdegree("cp.+ y degree", false, false);
  pangolin::Var<bool> minusYdegree("cp.- y degree", false, false);
  pangolin::Var<bool> addZdegree("cp.+ z degree", false, false);
  pangolin::Var<bool> minusZdegree("cp.- z degree", false, false);
  pangolin::Var<bool> addXtrans("cp.+ x trans", false, false);
  pangolin::Var<bool> minusXtrans("cp.- x trans", false, false);
  pangolin::Var<bool> addYtrans("cp.+ y trans", false, false);
  pangolin::Var<bool> minusYtrans("cp.- y trans", false, false);
  pangolin::Var<bool> addZtrans("cp.+ z trans", false, false);
  pangolin::Var<bool> minusZtrans("cp.- z trans", false, false);

  pangolin::Var<bool> resetButton("cp.Reset", false, false);
  pangolin::Var<bool> saveImg("cp.Save Result", false, false);

  std::vector<pangolin::Var<bool>> mat_calib_box;
  mat_calib_box.push_back(addXdegree);
  mat_calib_box.push_back(minusXdegree);
  mat_calib_box.push_back(addYdegree);
  mat_calib_box.push_back(minusYdegree);
  mat_calib_box.push_back(addZdegree);
  mat_calib_box.push_back(minusZdegree);
  mat_calib_box.push_back(addXtrans);
  mat_calib_box.push_back(minusXtrans);
  mat_calib_box.push_back(addYtrans);
  mat_calib_box.push_back(minusYtrans);
  mat_calib_box.push_back(addZtrans);
  mat_calib_box.push_back(minusZtrans);

  int frame_num = 0;
  int source_lidar_ptsize = source_cloud->points.size();
  int target_lidar_ptsize = target_cloud->points.size();
  ProcessTargetFrame(target_cloud, display_mode_);
  ProcessSourceFrame(source_cloud, calibration_matrix_, display_mode_);

  std::cout << "\n=>START\n";
  while (!pangolin::ShouldQuit()) {
    s_cam.Follow(Twc);
    d_cam.Activate(s_cam);



    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // 绘制坐标轴
    DrawCoordinateAxis();


    DrawTransformedAxis(calibration_matrix_, 3.0f); // 可调长度


    


    // DrawTransformedCoordinateAxis(calibration_matrix_, 5.5f);  // 可调整长度

    if (displayMode) {
      if (display_mode_ == false) {
        display_mode_ = true;
        ProcessTargetFrame(target_cloud, display_mode_);
        ProcessSourceFrame(source_cloud, calibration_matrix_, display_mode_);
      }
    } else {
      if (display_mode_ == true) {
        display_mode_ = false;
        ProcessTargetFrame(target_cloud, display_mode_);
        ProcessSourceFrame(source_cloud, calibration_matrix_, display_mode_);
      }
    }
    if (pointSize.GuiChanged()) {
      point_size_ = pointSize.Get();
      std::cout << "Point size changed to " << point_size_ << " degree\n";
    }

    if (degreeStep.GuiChanged()) {
      cali_scale_degree_ = degreeStep.Get();
      CalibrationScaleChange();
      std::cout << "Degree calib scale changed to " << cali_scale_degree_
                << " degree\n";
    }
    if (tStep.GuiChanged()) {
      cali_scale_trans_ = tStep.Get() / 100.0;
      CalibrationScaleChange();
      std::cout << "Trans calib scale changed to " << cali_scale_trans_ * 100
                << " cm\n";
    }
    for (int i = 0; i < 12; i++) {
      if (pangolin::Pushed(mat_calib_box[i])) {
        calibration_matrix_ = calibration_matrix_ * modification_list_[i];
        ProcessSourceFrame(source_cloud, calibration_matrix_, display_mode_);
        std::cout << "Changed!\n";
      }
    }

    if (pangolin::Pushed(resetButton)) {
      calibration_matrix_ = orign_calibration_matrix_;
      ProcessSourceFrame(source_cloud, calibration_matrix_, display_mode_);
      std::cout << "Reset!\n";
    }
    if (pangolin::Pushed(saveImg)) {
      std::string base_name = GetFileBaseName(extrinsic_json);
      std::string output_file_name = "output/" + base_name;
      saveResult(frame_num,output_file_name);
      std::cout << "\n==>Save Result " << frame_num << std::endl;
      Eigen::Matrix4d transform = calibration_matrix_;
      cout << "Transfromation Matrix:\n" << transform << std::endl;
      frame_num++;
    }

    if (kbhit()) {
      int c = getchar();
      if (ManualCalibration(c)) {
        Eigen::Matrix4d transform = calibration_matrix_;
        ProcessSourceFrame(source_cloud, calibration_matrix_, display_mode_);
        cout << "\nTransfromation Matrix:\n" << transform << std::endl;
      }
    }

    // draw lidar points
    glDisable(GL_LIGHTING);
    glPointSize(point_size_);
    // draw target lidar points
    target_colorBuffer_->Bind();
    glColorPointer(target_colorBuffer_->count_per_element,
                   target_colorBuffer_->datatype, 0, 0);
    glEnableClientState(GL_COLOR_ARRAY);
    target_vertexBuffer_->Bind();
    glVertexPointer(target_vertexBuffer_->count_per_element,
                    target_vertexBuffer_->datatype, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_POINTS, 0, target_lidar_ptsize);
    glDisableClientState(GL_VERTEX_ARRAY);
    target_vertexBuffer_->Unbind();
    glDisableClientState(GL_COLOR_ARRAY);
    target_colorBuffer_->Unbind();

    // draw source lidar points
    source_colorBuffer_->Bind();
    glColorPointer(source_colorBuffer_->count_per_element,
                   source_colorBuffer_->datatype, 0, 0);
    glEnableClientState(GL_COLOR_ARRAY);
    source_vertexBuffer_->Bind();
    glVertexPointer(source_vertexBuffer_->count_per_element,
                    source_vertexBuffer_->datatype, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_POINTS, 0, source_lidar_ptsize);
    glDisableClientState(GL_VERTEX_ARRAY);
    source_vertexBuffer_->Unbind();
    glDisableClientState(GL_COLOR_ARRAY);
    source_colorBuffer_->Unbind();

    pangolin::FinishFrame();
    usleep(100);
    glFinish();
  }

  // delete[] imageArray;

  Eigen::Matrix4d transform = calibration_matrix_;
  cout << "\nFinal Transfromation Matrix:\n" << transform << std::endl;

  return 0;
}
