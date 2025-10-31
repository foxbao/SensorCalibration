/*
 * Copyright (C) 2021 by Autonomous Driving Group, Shanghai AI Laboratory
 * Limited. All rights reserved.
 * Yan Guohang <yanguohang@pjlab.org.cn>
 * Ouyang Jinhua <ouyangjinhua@pjlab.org.cn>
 */
#pragma once

#include <fstream>
#include <iostream>
#include <jsoncpp/json/json.h>
#include <stdio.h>
#include <string>

// void LoadIntrinsic(const std::string &filename, Eigen::Matrix3d &K,
//                    std::vector<double> &dist) {
//   Json::Reader reader;
//   Json::Value root;

//   std::ifstream in(filename, std::ios::binary);

//   // std::ifstream in;
//   // in.open(filename);
//   if (!in.is_open()) {
//     std::cout << "Error Opening " << filename << std::endl;
//     return;
//   }
//   if (reader.parse(in, root, false)) {
//     Json::Value::Members name = root.getMemberNames();
//     std::string id = *(name.begin());
//     std::cout << id << std::endl;
//     Json::Value intri = root[id]["param"]["cam_K"]["data"];
//     Json::Value d = root[id]["param"]["cam_dist"]["data"];
//     int dist_col = root[id]["param"]["cam_dist"]["cols"].asInt();
//     K << intri[0][0].asDouble(), intri[0][1].asDouble(), intri[0][2].asDouble(),
//         intri[1][0].asDouble(), intri[1][1].asDouble(), intri[1][2].asDouble(),
//         intri[2][0].asDouble(), intri[2][1].asDouble(), intri[2][2].asDouble();
//     dist.clear();
//     for (int i = 0; i < dist_col; i++) {
//       dist.push_back(d[0][i].asDouble());
//     }
//   }
//   in.close();

//   return;
// }

void LoadIntrinsic(const std::string &filename, Eigen::Matrix3d &K,
                   std::vector<double> &dist, std::string &model) {
  Json::Reader reader;
  Json::Value root;

  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "Error Opening " << filename << std::endl;
    return;
  }
  if (reader.parse(in, root, false)) {
    Json::Value::Members name = root.getMemberNames();
    if (name.empty()) {
      std::cerr << "Intrinsic json has no top-level keys: " << filename
                << std::endl;
      in.close();
      return;
    }
    std::string id = *(name.begin());
    std::cout << "Load intrinsic id: " << id << std::endl;

    // read model if present
    if (root[id].isMember("param") && root[id]["param"].isMember("model")) {
      model = root[id]["param"]["model"].asString();
    } else {
      model = ""; // unknown -> treat as pinhole by default
    }

    Json::Value intri = root[id]["param"]["cam_K"]["data"];
    if (intri.size() >= 3 && intri[0].size() >= 3) {
      K << intri[0][0].asDouble(), intri[0][1].asDouble(),
          intri[0][2].asDouble(), intri[1][0].asDouble(),
          intri[1][1].asDouble(), intri[1][2].asDouble(),
          intri[2][0].asDouble(), intri[2][1].asDouble(),
          intri[2][2].asDouble();
    } else {
      std::cerr << "Invalid cam_K format in " << filename << std::endl;
    }

    // robustly read cam_dist: support [[a],[b],...] or [a,b,...]
    dist.clear();
    Json::Value d = root[id]["param"]["cam_dist"]["data"];
    if (d.isArray()) {
      // if d is like [[-0.06],[...],...]
      bool rows_are_arrays = (d.size() > 0 && d[0].isArray());
      if (rows_are_arrays) {
        for (Json::Value::ArrayIndex i = 0; i < d.size(); ++i) {
          if (d[i].isArray() && d[i].size() > 0) {
            // push all elements in row
            for (Json::Value::ArrayIndex j = 0; j < d[i].size(); ++j) {
              dist.push_back(d[i][j].asDouble());
            }
          }
        }
      } else {
        // d is [a, b, c]
        for (Json::Value::ArrayIndex i = 0; i < d.size(); ++i) {
          dist.push_back(d[i].asDouble());
        }
      }
    } else {
      // fallback: maybe cam_dist is stored differently (single column)
      try {
        // try to read as single column values
        Json::Value d2 = root[id]["param"]["cam_dist"];
        if (d2.isArray()) {
          for (Json::Value::ArrayIndex i = 0; i < d2.size(); ++i)
            dist.push_back(d2[i].asDouble());
        }
      } catch (...) {
        // give up silently
      }
    }
  } else {
    std::cerr << "Parse intrinsic json failed: " << filename << std::endl;
  }

  in.close();
  return;
}


// void LoadFishEyeIntrinsic(const std::string &filename, Eigen::Matrix3d &K,
//                    std::vector<double> &dist, std::string &model_type) {
//   Json::Reader reader;
//   Json::Value root;
//   std::ifstream in(filename, std::ios::binary);

//   if (!in.is_open()) {
//     std::cerr << "Error Opening " << filename << std::endl;
//     return;
//   }

//   if (reader.parse(in, root, false)) {
//     // 支持结构如 camera_xxx: { param: {...} }
//     Json::Value::Members name = root.getMemberNames();
//     std::string id = *(name.begin());
//     Json::Value intri = root[id]["param"]["cam_K"]["data"];
//     Json::Value d = root[id]["param"]["cam_dist"]["data"];
//     Json::Value model = root[id]["param"]["model"];
//     model_type = model.asString(); // <== 读取 "fisheye" 或 "pinhole"

//     // 兼容不同 JSON 中 distortion 结构（无 cols 时自动计算长度）
//     int dist_rows = d.size();
//     int dist_cols = d[0].size();

//     K << intri[0][0].asDouble(), intri[0][1].asDouble(), intri[0][2].asDouble(),
//         intri[1][0].asDouble(), intri[1][1].asDouble(), intri[1][2].asDouble(),
//         intri[2][0].asDouble(), intri[2][1].asDouble(), intri[2][2].asDouble();

//     dist.clear();
//     for (int i = 0; i < dist_rows; i++) {
//       for (int j = 0; j < dist_cols; j++) {
//         dist.push_back(d[i][j].asDouble());
//       }
//     }

//     std::cout << "Loaded camera model: " << model_type << std::endl;
//   }

//   in.close();
// }