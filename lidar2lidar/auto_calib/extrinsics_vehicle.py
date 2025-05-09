import numpy as np
from scipy.spatial.transform import Rotation as R
import json

def to_matrix(transform):
    t = np.array(transform[0:3])
    q = np.array(transform[3:7])
    rot = R.from_quat(q).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = t
    return T

def to_transform_vector(matrix):
    t = matrix[:3, 3]
    r = R.from_matrix(matrix[:3, :3]).as_quat()
    return np.concatenate([t, r])

def compute_and_save_extrinsic(
    extrinsics_file="extrinsics.json",
    relative_file="helios_front_helios_rear.json",
    output_file="computed_helios_rear_extrinsic.json"
):
    # 1. Load JSON files
    with open(extrinsics_file) as f1:
        extrinsics = json.load(f1)

    with open(relative_file) as f2:
        relative = json.load(f2)

    # 2. 获取前向外参向量和矩阵
    front_vec = extrinsics["Tx_baselink_lidar_helios_front_left"]
    T_base_front = to_matrix(front_vec)
    
    # 3. 获取前到后外参并转矩阵
    T_front_rear = to_matrix(relative["transform"])

    # 4. 计算 baselink → rear
    T_base_rear = T_base_front @ T_front_rear
    rear_vec = to_transform_vector(T_base_rear)

    # 5. 构建输出 JSON
    result = {
        "Tx_baselink_lidar_helios_front_left": front_vec,
        "Tx_baselink_lidar_helios_rear": rear_vec.tolist()
    }

    with open(output_file, "w") as fout:
        json.dump(result, fout, indent=4)

    print(f"✅ 外参写入成功：{output_file}")

def main():
    compute_and_save_extrinsic(
        extrinsics_file="extrinsics.json",
        relative_file="helios_front_helios_rear.json",
        output_file="computed_helios_rear_extrinsic.json"
    )

if __name__ == "__main__":
    main()
