#!/usr/bin/env python3
"""
Extract Ground-Truth from Rosbag
Combines /leica/position and /imu0 to create an approximate ground-truth CSV.
Note: This is not the official ground-truth but combines available data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation
import argparse


def extract_groundtruth_from_rosbag(rosbag_path, output_csv):
    """
    Extract ground-truth data from rosbag by combining:
    - /leica/position: Position (x, y, z)
    - /imu0: Orientation (quaternion)
    
    Args:
        rosbag_path: Path to rosbag directory
        output_csv: Output CSV file path
    """
    print(f"Reading rosbag from {rosbag_path}...")
    
    positions = {}
    orientations = {}
    
    with AnyReader([Path(rosbag_path)]) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == '/leica/position':
                msg = reader.deserialize(rawdata, connection.msgtype)
                positions[timestamp] = np.array([
                    msg.point.x,
                    msg.point.y,
                    msg.point.z
                ])
            
            elif connection.topic == '/imu0':
                msg = reader.deserialize(rawdata, connection.msgtype)
                orientations[timestamp] = np.array([
                    msg.orientation.w,
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z
                ])
    
    print(f"Found {len(positions)} position measurements")
    print(f"Found {len(orientations)} orientation measurements")
    
    all_timestamps = sorted(set(list(positions.keys()) + list(orientations.keys())))
    
    data = []
    for ts in all_timestamps:
        pos_ts = min(positions.keys(), key=lambda x: abs(x - ts)) if positions else None
        ori_ts = min(orientations.keys(), key=lambda x: abs(x - ts)) if orientations else None
        
        if pos_ts and abs(pos_ts - ts) < 1e9:
            pos = positions[pos_ts]
        else:
            pos = np.array([0.0, 0.0, 0.0])
        
        if ori_ts and abs(ori_ts - ts) < 1e9:
            quat = orientations[ori_ts]
        else:
            quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        data.append({
            'timestamp': ts,
            'p_x': pos[0],
            'p_y': pos[1],
            'p_z': pos[2],
            'q_w': quat[0],
            'q_x': quat[1],
            'q_y': quat[2],
            'q_z': quat[3]
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('timestamp')
    
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_csv, index=False, header=False)
    
    print(f"\nGround-truth CSV saved to {output_csv}")
    print(f"Total poses: {len(df)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    return output_csv


def main():
    parser = argparse.ArgumentParser(description='Extract ground-truth from rosbag')
    parser.add_argument('--rosbag', type=str, required=True, help='Path to rosbag directory')
    parser.add_argument('--output', type=str, default='data/MH_01_easy_groundtruth.csv',
                        help='Output CSV file')
    
    args = parser.parse_args()
    
    extract_groundtruth_from_rosbag(args.rosbag, args.output)


if __name__ == '__main__':
    main()


