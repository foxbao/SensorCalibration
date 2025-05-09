import numpy as np
import json
from collections import OrderedDict
import math

def quaternion_to_rotation_matrix(q):
    """将四元数转换为旋转矩阵 (w, x, y, z)顺序"""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,    2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y]
    ])

def rotation_matrix_to_euler_angles(R):
    """将旋转矩阵转换为欧拉角(弧度) Roll(X), Pitch(Y), Yaw(Z)"""
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2,1], R[2,2])
        pitch = math.atan2(-R[2,0], sy)
        yaw = math.atan2(R[1,0], R[0,0])
    else:
        roll = math.atan2(-R[1,2], R[1,1])
        pitch = math.atan2(-R[2,0], sy)
        yaw = 0
    
    return np.array([roll, pitch, yaw])

def transform_matrix_from_params(params):
    """从外参参数创建齐次变换矩阵"""
    tx, ty, tz, qx, qy, qz, qw = params
    rotation = quaternion_to_rotation_matrix([qw, qx, qy, qz])
    translation = np.array([tx, ty, tz])
    
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform

def inverse_transform_matrix(transform):
    """计算齐次变换矩阵的逆"""
    inv_transform = np.eye(4)
    inv_rotation = transform[:3, :3].T
    inv_translation = -inv_rotation @ transform[:3, 3]
    
    inv_transform[:3, :3] = inv_rotation
    inv_transform[:3, 3] = inv_translation
    return inv_transform

def compute_relative_transform(source_transform, target_transform):
    """计算从源传感器到目标传感器的相对变换"""
    # 相对变换 = inv(source) * target
    return inverse_transform_matrix(source_transform) @ target_transform

def transform_matrix_to_rpy_t(transform):
    """将齐次变换矩阵转换为RPY(度)和平移"""
    translation = transform[:3, 3]
    rotation = transform[:3, :3]
    
    roll, pitch, yaw = rotation_matrix_to_euler_angles(rotation)
    
    return [
        float(math.degrees(roll)),   # Roll in degrees
        float(math.degrees(pitch)),  # Pitch in degrees
        float(math.degrees(yaw)),    # Yaw in degrees
        float(translation[0]),       # tx in meters
        float(translation[1]),       # ty in meters
        float(translation[2])        # tz in meters
    ]

def compute_lidar_relative_transforms(extrinsics):
    """计算所有激光雷达之间的相对外参(RPY格式)"""
    # 传感器简称映射
    lidar_names = {
        "Tx_baselink_lidar_bp_front_left": "bp_fl",
        "Tx_baselink_lidar_bp_rear_right": "bp_rr",
        "Tx_baselink_lidar_helios_front_left": "helios_fl",
        "Tx_baselink_lidar_helios_rear_right": "helios_rr"
    }
    
    # 转换为变换矩阵
    transforms = {lidar_names[name]: transform_matrix_from_params(params) 
                 for name, params in extrinsics.items()}
    
    # 计算所有两两之间的相对变换
    relative_transforms = OrderedDict()
    sensor_ids = list(transforms.keys())
    
    for i, target in enumerate(sensor_ids):
        for j, source in enumerate(sensor_ids):
            if i != j:
                key = f"{target}_{source}"  # 目标_源 命名方式
                relative_transform = compute_relative_transform(transforms[source], transforms[target])
                rpy_t = transform_matrix_to_rpy_t(relative_transform)
                relative_transforms[key] = rpy_t
                
    return relative_transforms

def main():
    # 读取外参文件
    with open('extrinsics.json', 'r') as f:
        extrinsics = json.load(f)
    
    # 计算所有激光雷达相对外参(RPY格式)
    relative_transforms = compute_lidar_relative_transforms(extrinsics)
    
    # 保存结果到新文件
    with open('lidar_relative_extrinsics_rpy_target_source.json', 'w') as f:
        json.dump(relative_transforms, f, indent=4)
    
    print("激光雷达相对外参(RPY格式)计算完成，结果已保存到 lidar_relative_extrinsics_rpy_target_source.json")
    print("命名规则: 目标传感器_源传感器")
    print("数据格式: [Roll(°), Pitch(°), Yaw(°), tx(m), ty(m), tz(m)]")
    print(f"共计算了 {len(relative_transforms)} 个相对外参关系")

    # 打印示例
    print("\n示例结果:")
    example_key = next(iter(relative_transforms.keys()))
    print(f"{example_key}: {relative_transforms[example_key]}")

if __name__ == "__main__":
    main()