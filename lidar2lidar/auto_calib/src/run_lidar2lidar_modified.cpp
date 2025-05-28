#include <chrono> // NOLINT
#include <iostream>
#include <pcl/common/transforms.h>
#include <thread> // NOLINT
#include <time.h>
#include "nlohmann/json.hpp"

#include "calibration.hpp"
using json = nlohmann::json;

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

// void LoadPointCloud(
//     const std::string &filename,
//     std::map<int32_t, pcl::PointCloud<pcl::PointXYZI>> &lidar_points)
// {

//   std::ifstream file(filename);
//   if (!file.is_open())
//   {
//     std::cout << "[ERROR] open file " << filename << " failed." << std::endl;
//     exit(1);
//   }
//   std::string line, tmpStr;
//   while (getline(file, line))
//   {
//     int32_t device_id;
//     std::string point_cloud_path;
//     pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(
//         new pcl::PointCloud<pcl::PointXYZI>);

//     std::stringstream ss(line);
//     ss >> tmpStr >> device_id;
//     getline(file, line);
//     ss = std::stringstream(line);
//     ss >> tmpStr >> point_cloud_path;
//     if (pcl::io::loadPCDFile(point_cloud_path, *cloud) < 0)
//     {
//       std::cout << "[ERROR] cannot open pcd_file: " << point_cloud_path << "\n";
//       exit(1);
//     }
//     lidar_points.insert(std::make_pair(device_id, *cloud));
//   }
// }


void LoadPointCloud(
    const std::string &filename,
    std::map<int32_t, pcl::PointCloud<pcl::PointXYZI>> &lidar_points,
    double filter_threshold=-1.0)  // 加入 filter_threshold 参数
{
  std::ifstream file(filename);
  if (!file.is_open())
  {
    std::cout << "[ERROR] open file " << filename << " failed." << std::endl;
    exit(1);
  }

  std::string line, tmpStr;
  while (getline(file, line))
  {
    int32_t device_id;
    std::string point_cloud_path;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);

    std::stringstream ss(line);
    ss >> tmpStr >> device_id;

    getline(file, line);
    ss = std::stringstream(line);
    ss >> tmpStr >> point_cloud_path;

    if (pcl::io::loadPCDFile(point_cloud_path, *cloud) < 0)
    {
      std::cout << "[ERROR] cannot open pcd_file: " << point_cloud_path << "\n";
      exit(1);
    }

    // 👉 若为 device_id == 1 且启用了过滤（threshold > 0），则进行筛选
    if (device_id == 1 && filter_threshold > 0.0)
    {
      pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>);
      for (const auto &pt : cloud->points)
      {
        double distance = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
        if (distance >= filter_threshold)
        {
          filtered_cloud->points.push_back(pt);
        }
      }
      filtered_cloud->width = filtered_cloud->points.size();
      filtered_cloud->height = 1;
      filtered_cloud->is_dense = true;
      lidar_points.insert(std::make_pair(device_id, *filtered_cloud));
    }
    else
    {
      lidar_points.insert(std::make_pair(device_id, *cloud));
    }
  }
}


void LoadCalibFile(const std::string &filename,
                   std::map<int32_t, InitialExtrinsic> &calib_extrinsic)
{
  std::ifstream file(filename);
  if (!file.is_open())
  {
    std::cout << "open file " << filename << " failed." << std::endl;
    exit(1);
  }
  float degree_2_radian = 0.017453293;
  std::string line, tmpStr;
  while (getline(file, line))
  {
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

void SaveCalibrationResultsToJson(const std::map<int32_t, Eigen::Matrix4d> &refined_extrinsics,
                                  const std::string &output_file)
{
  json j;
  for (const auto &entry : refined_extrinsics)
  {
    // int32_t device_id = entry.first;  // 不再使用 device_id 作为 key
    const Eigen::Matrix4d &T = entry.second;

    // 提取平移
    Eigen::Vector3d translation = T.block<3, 1>(0, 3);

    // 提取旋转矩阵并转换为四元数
    Eigen::Matrix3d rotation = T.block<3, 3>(0, 0);
    Eigen::Quaterniond q(rotation);

    // 构造 transform: [tx, ty, tz, qx, qy, qz, qw]
    std::vector<double> transform = {
        translation.x(), translation.y(), translation.z(),
        q.x(), q.y(), q.z(), q.w()};

    // 直接存储 transform，不嵌套在 device_id 下
    j["transform"] = transform;
    break; // 如果只需要第一个设备的变换，就 break；否则需要调整结构
  }

  std::ofstream o(output_file);
  if (!o.is_open())
  {
    std::cerr << "[ERROR] Could not write JSON to " << output_file << std::endl;
    return;
  }

  o << j.dump(4); // pretty print with indent=4
  std::cout << "Saved calibration to JSON file: " << output_file << std::endl;
}

void SaveCalibrationResults(const std::map<int32_t, Eigen::Matrix4d> &refined_extrinsics,
                            const std::string &output_file)
{
  std::ofstream outfile(output_file);
  if (!outfile.is_open())
  {
    std::cerr << "Error: Could not open output file " << output_file << std::endl;
    return;
  }

  for (const auto &entry : refined_extrinsics)
  {
    int32_t device_id = entry.first;
    const Eigen::Matrix4d &transform = entry.second;

    outfile << "device_id: " << device_id << std::endl;
    outfile << "transformation matrix:" << std::endl;
    outfile << transform << std::endl;
    outfile << "-----------------------------" << std::endl;
  }

  outfile.close();
  std::cout << "Calibration results saved to " << output_file << std::endl;
}

int main(int argc, char *argv[])
{

  int32_t device_id = 18;
  Eigen::Matrix4d init_ext_input = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d curr_transform = Eigen::Matrix4d::Identity();
  // std::map<int32_t, Eigen::Matrix4d> init_extrinsics_;
  std::map<int32_t, Eigen::Matrix4d, std::less<int32_t>, Eigen::aligned_allocator<std::pair<const int32_t, Eigen::Matrix4d>>> init_extrinsics_;

  init_extrinsics_.insert(std::make_pair(device_id, init_ext_input));
  int32_t slave_id = 18;
  Eigen::Matrix4d T_ms = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d init_ext = init_extrinsics_[slave_id];

  curr_transform = init_ext * T_ms;

  // Eigen::Matrix4d curr_transform = Eigen::Matrix4d::Identity();
  // curr_transform = init_ext * T_ms;
  std::string filename = "calibration_results.json";
  double filter_threshold = -1.0; // 默认：-1 表示不启用过滤
  if (argc < 3)
  {
    std::cout << "Usage: ./run_lidar2lidar <lidar_file> <calib_file>"
                 "\nexample:\n\t"
                 "./bin/run_lidar2lidar data/0001/lidar_cloud_path.txt "
                 "data/0001/initial_extrinsic.txt"
              << std::endl;
    return 0;
  }
  auto lidar_file = argv[1];
  auto calib_file = argv[2];
  if (argc >= 4)
  {
    filename = argv[3]; // 第4个参数（索引3）作为文件名
  }
  if (argc >= 5)
  {
    filter_threshold = std::stod(argv[4]); // 第五个参数转换为 double
  }
  std::map<int32_t, pcl::PointCloud<pcl::PointXYZI>> lidar_points;
  LoadPointCloud(lidar_file, lidar_points,filter_threshold);
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

  // Save calibration results to file
  SaveCalibrationResults(refined_extrinsics, "calibration_results.txt");
  SaveCalibrationResultsToJson(refined_extrinsics, filename);

  // stitching
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr all_cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  auto master_iter = lidar_points.find(0);
  pcl::PointCloud<pcl::PointXYZI> master_pc = master_iter->second;
  for (auto src : master_pc.points)
  {
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
       iter++)
  {
    int32_t slave_id = iter->first;
    Eigen::Matrix4d transform = iter->second;

    auto slave_iter = lidar_points.find(slave_id);
    pcl::PointCloud<pcl::PointXYZI> slave_pc = slave_iter->second;

    pcl::PointCloud<pcl::PointXYZI> trans_cloud;
    pcl::transformPointCloud(slave_pc, trans_cloud, transform);
    for (auto src : trans_cloud.points)
    {
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
  return 0;
}
