import numpy as np
import json
from scipy.spatial.transform import Rotation as R

def load_transform_from_json(file_path, key=None):
    """从JSON文件中加载平移和四元数（xyzw格式）"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    if key is not None:
        transform_data = data[key]
    else:
        transform_data = data["transform"]
    
    translation = np.array(transform_data[:3])
    quaternion_xyzw = np.array(transform_data[3:])
    return translation, quaternion_xyzw

def transform_to_matrix(translation, quaternion_xyzw):
    """将平移和四元数转换为4x4齐次变换矩阵"""
    rotation = R.from_quat(quaternion_xyzw).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform

def matrix_to_transform(matrix):
    """从4x4齐次变换矩阵提取平移和四元数（xyzw格式）"""
    translation = matrix[:3, 3]
    rotation = R.from_matrix(matrix[:3, :3]).as_quat()  # xyzw格式
    return translation, rotation

def compose_transforms(transform1, transform2):
    """组合两个4x4齐次变换矩阵：T1 * T2"""
    return np.dot(transform1, transform2)

def save_to_json(data, file_path):
    """将数据保存为JSON文件"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def calculate_transforms():
    """计算所有目标外参"""
    # 文件路径
    extrinsics_file = "extrinsics.json"
    helios_front_bp_front_file = "helios_front_bp_front.json"
    helios_front_helios_rear_file = "helios_front_helios_rear.json"
    helios_rear_bp_rear_file = "helios_rear_bp_rear.json"
    output_file = "output.json"

    # 1. 加载所有外参并转换为变换矩阵
    # heilos_front到baselink的外参
    t_helios_front_to_baselink, q_helios_front_to_baselink = load_transform_from_json(
        extrinsics_file, "Tx_baselink_lidar_helios_front_left"
    )
    T_helios_front_to_baselink = transform_to_matrix(t_helios_front_to_baselink, q_helios_front_to_baselink)

    # bp_front到helios_front的外参
    t_bp_front_to_helios_front, q_bp_front_to_helios_front = load_transform_from_json(helios_front_bp_front_file)
    T_bp_front_to_helios_front = transform_to_matrix(t_bp_front_to_helios_front, q_bp_front_to_helios_front)

    # helios_rear到helios_front的外参
    t_helios_rear_to_helios_front, q_helios_rear_to_helios_front = load_transform_from_json(helios_front_helios_rear_file)
    T_helios_rear_to_helios_front = transform_to_matrix(t_helios_rear_to_helios_front, q_helios_rear_to_helios_front)

    # bp_rear到helios_rear的外参
    t_bp_rear_to_helios_rear, q_bp_rear_to_helios_rear = load_transform_from_json(helios_rear_bp_rear_file)
    T_bp_rear_to_helios_rear = transform_to_matrix(t_bp_rear_to_helios_rear, q_bp_rear_to_helios_rear)

    # 2. 计算目标外参
    # (1) helios_front到baselink的外参（直接使用输入数据）
    t_helios_front_to_baselink = t_helios_front_to_baselink
    q_helios_front_to_baselink = q_helios_front_to_baselink

    # (2) helios_rear到baselink的外参
    T_helios_rear_to_baselink = compose_transforms(T_helios_front_to_baselink, T_helios_rear_to_helios_front)
    t_helios_rear_to_baselink, q_helios_rear_to_baselink = matrix_to_transform(T_helios_rear_to_baselink)

    # (3) bp_front到baselink的外参
    T_bp_front_to_baselink = compose_transforms(T_helios_front_to_baselink, T_bp_front_to_helios_front)
    t_bp_front_to_baselink, q_bp_front_to_baselink = matrix_to_transform(T_bp_front_to_baselink)

    # (4) bp_rear到baselink的外参
    T_bp_rear_to_baselink = compose_transforms(
        compose_transforms(T_helios_front_to_baselink, T_helios_rear_to_helios_front),
        T_bp_rear_to_helios_rear
    )
    t_bp_rear_to_baselink, q_bp_rear_to_baselink = matrix_to_transform(T_bp_rear_to_baselink)

    # 3. 保存结果
    result = {
        "Tx_baselink_lidar_helios_front_left": list(np.concatenate([t_helios_front_to_baselink, q_helios_front_to_baselink])),
        "Tx_baselink_lidar_helios_rear_right": list(np.concatenate([t_helios_rear_to_baselink, q_helios_rear_to_baselink])),
        "Tx_baselink_lidar_bp_front_left": list(np.concatenate([t_bp_front_to_baselink, q_bp_front_to_baselink])),
        "Tx_baselink_lidar_bp_rear_right": list(np.concatenate([t_bp_rear_to_baselink, q_bp_rear_to_baselink]))
    }

    save_to_json(result, output_file)
    print(f"计算结果已保存到 {output_file}")

def main():
    """主函数"""
    try:
        calculate_transforms()
        print("计算完成！")
    except Exception as e:
        print(f"计算过程中发生错误：{e}")

if __name__ == "__main__":
    main()