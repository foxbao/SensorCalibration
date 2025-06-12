#include <chrono> // NOLINT
#include <iostream>
#include <pcl/common/transforms.h>
#include <thread> // NOLINT
#include <time.h>
#include <fstream>
#include <filesystem>
#include "calibration.hpp"
#include <nlohmann/json.hpp>
#include <cmath>  // M_PI, std::atan2, std::asin


Eigen::Vector3d RotationMatrixToEulerDegrees(const Eigen::Matrix3d& R) {
  double roll, pitch, yaw;

  pitch = std::asin(-R(2, 0));
  if (std::abs(R(2, 0)) < 0.9999) {
    roll = std::atan2(R(2, 1), R(2, 2));
    yaw = std::atan2(R(1, 0), R(0, 0));
  } else {
    // Gimbal lock
    roll = 0.0;
    yaw = std::atan2(-R(0, 1), R(1, 1));
  }

  // 转度
  roll *= 180.0 / M_PI;
  pitch *= 180.0 / M_PI;
  yaw *= 180.0 / M_PI;

  return Eigen::Vector3d(roll, pitch, yaw);
}

void SaveStitchedPointCloud(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
    const std::string& input_lidar_path) {

  namespace fs = std::filesystem;
  fs::path input_file_path(input_lidar_path);
  fs::path relative_path = input_file_path.lexically_relative("data");

  // 提取目录层级
  std::vector<std::string> parts;
  for (const auto& part : relative_path) {
    parts.push_back(part.string());
  }

  if (parts.size() < 2) {
    std::cerr << "Invalid input path for saving point cloud" << std::endl;
    return;
  }

  std::string scene_id = parts[0];
  std::string sensor_pair = parts[1];

  fs::path output_dir = fs::path("output") / scene_id;
  fs::create_directories(output_dir);

  fs::path pcd_file = output_dir / (sensor_pair + ".pcd");

  if (pcl::io::savePCDFileBinary(pcd_file.string(), *cloud) == 0) {
    std::cout << "✅ Saved stitched point cloud to: " << pcd_file << std::endl;
  } else {
    std::cerr << "Failed to save stitched point cloud to: " << pcd_file << std::endl;
  }
}
void SaveExtrinsicsToTxtAndJson(const std::map<int32_t, Eigen::Matrix4d>& refined_extrinsics,
                               const std::string& input_path) {
  namespace fs = std::filesystem;

  fs::path input_file_path(input_path);
  fs::path relative_path = input_file_path.lexically_relative("data");

  std::vector<std::string> parts;
  for (const auto& part : relative_path) {
    parts.push_back(part.string());
  }

  if (parts.size() < 2) {
    std::cerr << "Invalid input path, must include at least two components under data/" << std::endl;
    return;
  }

  std::string scene_id = parts[0];
  std::string sensor_pair = parts[1];

  fs::path output_dir = fs::path("output") / scene_id;
  fs::create_directories(output_dir);

  fs::path txt_file = output_dir / (sensor_pair + ".txt");
  fs::path json_file = output_dir / (sensor_pair + ".json");

  std::ofstream txt_ofs(txt_file);
  if (!txt_ofs.is_open()) {
    std::cerr << "Failed to open TXT file: " << txt_file << std::endl;
    return;
  }

  nlohmann::json json_out;

  for (const auto& [lidar_id, matrix] : refined_extrinsics) {
    txt_ofs << "Lidar ID: " << lidar_id << "\n";

    // 4x4矩阵格式化输出
    txt_ofs << "Transform matrix (4x4):\n";
    for (int i = 0; i < 4; ++i) {
      txt_ofs << "[";
      for (int j = 0; j < 4; ++j) {
        txt_ofs << matrix(i, j);
        if (j < 3) txt_ofs << ",";
      }
      txt_ofs << "]";
      if (i < 3) txt_ofs << ",";
      txt_ofs << "\n";
    }
    txt_ofs << "\n";

    // 平移向量
    Eigen::Vector3d t = matrix.block<3, 1>(0, 3);
    txt_ofs << "Translation (tx, ty, tz):\n";
    txt_ofs << t.transpose() << "\n\n";

    // 欧拉角
    Eigen::Matrix3d R = matrix.block<3, 3>(0, 0);
    Eigen::Vector3d euler_deg = RotationMatrixToEulerDegrees(R);
    txt_ofs << "Euler angles (roll, pitch, yaw in degrees):\n";
    txt_ofs << euler_deg.transpose() << "\n\n";

    // 四元数
    Eigen::Quaterniond q(R);
    txt_ofs << "Quaternion (qx, qy, qz, qw):\n";
    txt_ofs << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n\n";

    txt_ofs << "----------------------------------------\n";

    // json部分保留之前格式
    // json_out["Tx_" + sensor_pair + "_" + std::to_string(lidar_id)] = {t[0], t[1], t[2], q.x(), q.y(), q.z(), q.w()};
    json_out["Tx_" + sensor_pair] = {t[0], t[1], t[2], q.x(), q.y(), q.z(), q.w()};

  }

  txt_ofs.close();
  std::cout << "✅ Saved refined extrinsics to: " << txt_file << std::endl;

  std::ofstream json_ofs(json_file);
  if (!json_ofs.is_open()) {
    std::cerr << "Failed to open JSON file: " << json_file << std::endl;
    return;
  }

  json_ofs << json_out.dump(4);
  json_ofs.close();
  std::cout << "✅ Saved transformation parameters to: " << json_file << std::endl;
}

unsigned char color_map[10][3] = {{255, 255, 255}, // "white"
                                  {255, 0, 0},     // "red"
                                  {0, 255, 0},     // "green"
                                  {0, 0, 255},     // "blue"
                                  {255, 255, 0},   // "yellow"
                                  {255, 0, 255},   // "pink"
                                  {50, 255, 255},  // "light-blue"
                                  {135, 60, 0},    //
                                  {150, 240, 80},  //
                                  {80, 30, 180}};  //

void LoadPointCloud(
    const std::string &filename,
    std::map<int32_t, pcl::PointCloud<pcl::PointXYZI>> &lidar_points) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cout << "[ERROR] open file " << filename << " failed." << std::endl;
    exit(1);
  }
  std::string line, tmpStr;
  while (getline(file, line)) {
    int32_t device_id;
    std::string point_cloud_path;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZI>);

    std::stringstream ss(line);
    ss >> tmpStr >> device_id;
    getline(file, line);
    ss = std::stringstream(line);
    ss >> tmpStr >> point_cloud_path;
    if (pcl::io::loadPCDFile(point_cloud_path, *cloud) < 0) {
      std::cout << "[ERROR] cannot open pcd_file: " << point_cloud_path << "\n";
      exit(1);
    }
    lidar_points.insert(std::make_pair(device_id, *cloud));
  }
}

void LoadCalibFile(const std::string &filename,
                   std::map<int32_t, InitialExtrinsic> &calib_extrinsic) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cout << "open file " << filename << " failed." << std::endl;
    exit(1);
  }
  float degree_2_radian = 0.017453293;
  std::string line, tmpStr;
  while (getline(file, line)) {
    int32_t device_id;
    InitialExtrinsic extrinsic;
    std::stringstream ss(line);
    ss >> tmpStr >> device_id;
    getline(file, line);
    ss = std::stringstream(line);
    ss >> tmpStr >> extrinsic.euler_angles[0] >> extrinsic.euler_angles[1] >>
        extrinsic.euler_angles[2] >> extrinsic.t_matrix[0] >>
        extrinsic.t_matrix[1] >> extrinsic.t_matrix[2];

    extrinsic.euler_angles[0] = extrinsic.euler_angles[0] * degree_2_radian;
    extrinsic.euler_angles[1] = extrinsic.euler_angles[1] * degree_2_radian;
    extrinsic.euler_angles[2] = extrinsic.euler_angles[2] * degree_2_radian;
    calib_extrinsic.insert(std::make_pair(device_id, extrinsic));
  }
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cout << "Usage: ./run_lidar2lidar <lidar_file> <calib_file>"
                 "\nexample:\n\t"
                 "./bin/run_lidar2lidar data/0001/lidar_cloud_path.txt "
                 "data/0001/initial_extrinsic.txt"
              << std::endl;
    return 0;
  }
  auto lidar_file = argv[1];
  auto calib_file = argv[2];
  std::map<int32_t, pcl::PointCloud<pcl::PointXYZI>> lidar_points;
  LoadPointCloud(lidar_file, lidar_points);
  std::map<int32_t, InitialExtrinsic> extrinsics;
  LoadCalibFile(calib_file, extrinsics);

  // calibration
  Calibrator calibrator;
  calibrator.LoadCalibrationData(lidar_points, extrinsics);
  auto time_begin = std::chrono::steady_clock::now();
  calibrator.Calibrate();
  auto time_end = std::chrono::steady_clock::now();
  std::cout << "calib cost "
            << std::chrono::duration<double>(time_end - time_begin).count()
            << "s" << std::endl;
  std::map<int32_t, Eigen::Matrix4d> refined_extrinsics =
      calibrator.GetFinalTransformation();

  // 保存标定结果
  // SaveExtrinsicsToTxt(refined_extrinsics, argv[1]);  // argv[1] 是 lidar_cloud_path.txt 的路径
  SaveExtrinsicsToTxtAndJson(refined_extrinsics, argv[1]);


  // stitching
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr all_cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  auto master_iter = lidar_points.find(0);
  pcl::PointCloud<pcl::PointXYZI> master_pc = master_iter->second;
  for (auto src : master_pc.points) {
    int32_t master_id = 0;
    pcl::PointXYZRGB point;
    point.x = src.x;
    point.y = src.y;
    point.z = src.z;
    point.r = color_map[master_id % 7][0];
    point.g = color_map[master_id % 7][1];
    point.b = color_map[master_id % 7][2];
    all_cloud->push_back(point);
  }

  for (auto iter = refined_extrinsics.begin(); iter != refined_extrinsics.end();
       iter++) {
    int32_t slave_id = iter->first;
    Eigen::Matrix4d transform = iter->second;

    auto slave_iter = lidar_points.find(slave_id);
    pcl::PointCloud<pcl::PointXYZI> slave_pc = slave_iter->second;

    pcl::PointCloud<pcl::PointXYZI> trans_cloud;
    pcl::transformPointCloud(slave_pc, trans_cloud, transform);
    for (auto src : trans_cloud.points) {
      pcl::PointXYZRGB point;
      point.x = src.x;
      point.y = src.y;
      point.z = src.z;
      point.r = color_map[slave_id % 7][0];
      point.g = color_map[slave_id % 7][1];
      point.b = color_map[slave_id % 7][2];
      all_cloud->push_back(point);
    }
  }
  all_cloud->height = 1;
  all_cloud->width = all_cloud->points.size();
  std::string path = "stitching.pcd";
  pcl::io::savePCDFileBinary(path, *all_cloud);
  SaveStitchedPointCloud(all_cloud, argv[1]);
  return 0;
}