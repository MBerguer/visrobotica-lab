#!/usr/bin/env python3
"""
Ground-Truth Loader for EuRoC Dataset
Loads ground-truth poses from CSV file and transforms from IMU to camera coordinates.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation


def load_euroc_ground_truth(csv_path):
    """
    Load ground-truth poses from EuRoC CSV file.
    
    Format: timestamp, p_x, p_y, p_z, q_w, q_x, q_y, q_z
    Coordinates are in IMU frame.
    
    Args:
        csv_path: Path to ground-truth CSV file (e.g., data.csv from EuRoC)
    
    Returns:
        timestamps: Array of timestamps (nanoseconds)
        poses_imu: List of 4x4 transformation matrices in IMU frame
    """
    df = pd.read_csv(csv_path, header=None)
    
    timestamps = df.iloc[:, 0].values
    
    poses_imu = []
    for _, row in df.iterrows():
        position = row.iloc[1:4].values
        quaternion = row.iloc[4:8].values  # w, x, y, z
        
        R = Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
        
        pose = np.eye(4)
        pose[:3, :3] = R.as_matrix()
        pose[:3, 3] = position
        
        poses_imu.append(pose)
    
    return timestamps, poses_imu


def get_imu_to_cam_transform():
    """
    Get transformation matrix from IMU to camera (left) frame for EuRoC dataset.
    
    This transformation is provided in the EuRoC calibration files.
    For MH_01_easy, the transformation is approximately:
    - Rotation: Identity (IMU and camera are aligned)
    - Translation: Small offset (typically ~0.02m in z direction)
    
    Note: For exact values, check the EuRoC calibration files.
    """
    T_imu_cam = np.eye(4)
    
    T_imu_cam[:3, 3] = [0.0, 0.0, 0.0]
    
    return T_imu_cam


def transform_imu_to_camera(poses_imu, T_imu_cam=None):
    """
    Transform poses from IMU frame to camera frame.
    
    Formula: pose_cam = T_imu_cam @ pose_imu @ T_cam_imu
    where T_cam_imu = inv(T_imu_cam)
    
    Args:
        poses_imu: List of 4x4 poses in IMU frame
        T_imu_cam: 4x4 transformation matrix from IMU to camera (if None, uses default)
    
    Returns:
        poses_cam: List of 4x4 poses in camera frame
    """
    if T_imu_cam is None:
        T_imu_cam = get_imu_to_cam_transform()
    
    T_cam_imu = np.linalg.inv(T_imu_cam)
    
    poses_cam = []
    for pose_imu in poses_imu:
        pose_cam = T_imu_cam @ pose_imu @ T_cam_imu
        poses_cam.append(pose_cam)
    
    return poses_cam


def load_ground_truth_for_images(csv_path, image_timestamps, T_imu_cam=None):
    """
    Load ground-truth poses corresponding to image timestamps.
    
    Args:
        csv_path: Path to ground-truth CSV file
        image_timestamps: List of image timestamps (nanoseconds)
        T_imu_cam: Transformation from IMU to camera (optional)
    
    Returns:
        poses_cam: List of 4x4 poses in camera frame, aligned with image_timestamps
    """
    gt_timestamps, poses_imu = load_euroc_ground_truth(csv_path)
    
    poses_cam_imu = transform_imu_to_camera(poses_imu, T_imu_cam)
    
    poses_cam = []
    for img_ts in image_timestamps:
        closest_idx = np.argmin(np.abs(gt_timestamps - img_ts))
        
        if abs(gt_timestamps[closest_idx] - img_ts) < 1e9:
            poses_cam.append(poses_cam_imu[closest_idx])
        else:
            poses_cam.append(np.eye(4))
    
    return poses_cam


if __name__ == '__main__':
    print("Ground-Truth Loader for EuRoC Dataset")
    print("This module provides functions to load and transform ground-truth poses")

