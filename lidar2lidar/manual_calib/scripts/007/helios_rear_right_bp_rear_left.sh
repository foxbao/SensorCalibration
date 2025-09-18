#!/bin/bash
# 指定场景号
SCENE_ID=007
# 指定配置文件路径
CONFIG_FILE="cfgs/${SCENE_ID}.yaml"
PCD1=$(yq '.helios_rear_right_pcd' "$CONFIG_FILE")
PCD2=$(yq '.bp_rear_left_pcd' "$CONFIG_FILE")

INIT_FILE=$(yq '.helios_rear_right_bp_rear_left_init_file' "$CONFIG_FILE")

echo "PCD1 is: $PCD1"
echo "PCD2 is: $PCD2"
echo "INIT_FILE is: $INIT_FILE"


./bin/run_lidar2lidar "$PCD1" "$PCD2" "$INIT_FILE" "$SCENE_ID"