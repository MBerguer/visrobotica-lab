#!/usr/bin/env python3
"""
Trajectory Estimation using Monocular Vision
Exercise 2j: Estimate and visualize camera trajectory
"""

import cv2
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation


class TrajectoryEstimator:
    def __init__(self, calibration_file):
        with open(calibration_file, 'rb') as f:
            self.calib = pickle.load(f)
        
        self.K = self.calib['K_left']
        self.detector = cv2.ORB_create(nfeatures=2000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def extract_features(self, image):
        """Extract features from image."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2):
        """Match features between consecutive frames."""
        matches = self.bf_matcher.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches
    
    def estimate_pose(self, kp1, kp2, matches, scale=1.0):
        """Estimate relative pose between two frames."""
        if len(matches) < 8:
            return None, None, None
        
        points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        points2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        E, mask = cv2.findEssentialMat(
            points1, points2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        
        if E is None:
            return None, None, None
        
        _, R, t, mask_pose = cv2.recoverPose(E, points1, points2, self.K, mask=mask)
        
        t_scaled = t * scale
        
        return R, t_scaled, mask_pose
    
    def estimate_trajectory(self, images, gt_poses=None):
        """Estimate full trajectory from image sequence."""
        poses = [np.eye(4)]
        current_pose = np.eye(4)
        
        prev_kp, prev_desc = self.extract_features(images[0])
        
        for i in range(1, len(images)):
            print(f"Processing frame {i}/{len(images)-1}...")
            
            curr_kp, curr_desc = self.extract_features(images[i])
            
            if prev_desc is None or curr_desc is None or len(prev_desc) < 8 or len(curr_desc) < 8:
                poses.append(current_pose.copy())
                prev_kp, prev_desc = curr_kp, curr_desc
                continue
            
            matches = self.match_features(prev_desc, curr_desc)
            
            if len(matches) < 8:
                poses.append(current_pose.copy())
                prev_kp, prev_desc = curr_kp, curr_desc
                continue
            
            scale = 1.0
            if gt_poses is not None and i < len(gt_poses):
                gt_trans_prev = gt_poses[i-1][:3, 3]
                gt_trans_curr = gt_poses[i][:3, 3]
                scale = np.linalg.norm(gt_trans_curr - gt_trans_prev)
            
            R, t, mask = self.estimate_pose(prev_kp, curr_kp, matches, scale)
            
            if R is not None and t is not None:
                T_rel = np.eye(4)
                T_rel[:3, :3] = R
                T_rel[:3, 3:4] = t
                
                current_pose = current_pose @ T_rel
            
            poses.append(current_pose.copy())
            
            prev_kp, prev_desc = curr_kp, curr_desc
        
        return poses
    
    def visualize_trajectory(self, estimated_poses, gt_poses=None, output_path=None, show_axes=True, axes_scale=0.1):
        """Visualize estimated and ground-truth trajectories with camera coordinate systems."""
        est_positions = np.array([pose[:3, 3] for pose in estimated_poses])
        
        fig = plt.figure(figsize=(15, 10))
        
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot(est_positions[:, 0], est_positions[:, 1], est_positions[:, 2],
                'b-', linewidth=2, label='Estimated')
        if gt_poses is not None:
            gt_positions = np.array([pose[:3, 3] for pose in gt_poses])
            ax1.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2],
                    'r--', linewidth=2, label='Ground Truth')
        
        if show_axes:
            step = max(1, len(estimated_poses) // 20)
            for i in range(0, len(estimated_poses), step):
                pose = estimated_poses[i]
                pos = pose[:3, 3]
                R = pose[:3, :3]
                
                x_axis = R @ np.array([axes_scale, 0, 0])
                y_axis = R @ np.array([0, axes_scale, 0])
                z_axis = R @ np.array([0, 0, axes_scale])
                
                ax1.plot([pos[0], pos[0] + x_axis[0]], 
                        [pos[1], pos[1] + x_axis[1]], 
                        [pos[2], pos[2] + x_axis[2]], 'r-', linewidth=1, alpha=0.5)
                ax1.plot([pos[0], pos[0] + y_axis[0]], 
                        [pos[1], pos[1] + y_axis[1]], 
                        [pos[2], pos[2] + y_axis[2]], 'g-', linewidth=1, alpha=0.5)
                ax1.plot([pos[0], pos[0] + z_axis[0]], 
                        [pos[1], pos[1] + z_axis[1]], 
                        [pos[2], pos[2] + z_axis[2]], 'b-', linewidth=1, alpha=0.5)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory with Camera Axes')
        ax1.legend()
        ax1.grid(True)
        
        ax2 = fig.add_subplot(222)
        ax2.plot(est_positions[:, 0], est_positions[:, 1], 'b-', linewidth=2, label='Estimated')
        if gt_poses is not None:
            ax2.plot(gt_positions[:, 0], gt_positions[:, 1], 'r--', linewidth=2, label='Ground Truth')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Top View (X-Y)')
        ax2.legend()
        ax2.grid(True)
        ax2.axis('equal')
        
        ax3 = fig.add_subplot(223)
        ax3.plot(est_positions[:, 0], est_positions[:, 2], 'b-', linewidth=2, label='Estimated')
        if gt_poses is not None:
            ax3.plot(gt_positions[:, 0], gt_positions[:, 2], 'r--', linewidth=2, label='Ground Truth')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Z (m)')
        ax3.set_title('Side View (X-Z)')
        ax3.legend()
        ax3.grid(True)
        
        ax4 = fig.add_subplot(224)
        ax4.plot(est_positions[:, 1], est_positions[:, 2], 'b-', linewidth=2, label='Estimated')
        if gt_poses is not None:
            ax4.plot(gt_positions[:, 1], gt_positions[:, 2], 'r--', linewidth=2, label='Ground Truth')
        ax4.set_xlabel('Y (m)')
        ax4.set_ylabel('Z (m)')
        ax4.set_title('Side View (Y-Z)')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def compute_trajectory_error(self, estimated_poses, gt_poses):
        """Compute trajectory error metrics."""
        est_positions = np.array([pose[:3, 3] for pose in estimated_poses])
        gt_positions = np.array([pose[:3, 3] for pose in gt_poses])
        
        min_len = min(len(est_positions), len(gt_positions))
        est_positions = est_positions[:min_len]
        gt_positions = gt_positions[:min_len]
        
        errors = np.linalg.norm(est_positions - gt_positions, axis=1)
        
        ate = np.mean(errors)
        rmse = np.sqrt(np.mean(errors**2))
        max_error = np.max(errors)
        
        return {
            'ATE': ate,
            'RMSE': rmse,
            'Max Error': max_error,
            'Per-frame errors': errors
        }


def load_images_from_rosbag(rosbag_path, max_frames=200, skip_frames=5):
    """Load image sequence from rosbag."""
    images = []
    timestamps = []
    
    with AnyReader([Path(rosbag_path)]) as reader:
        frame_count = 0
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic != '/cam0/image_raw':
                continue
                
            if frame_count % skip_frames != 0:
                frame_count += 1
                continue
            
            msg = reader.deserialize(rawdata, connection.msgtype)
            
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, -1
            )
            
            if img.shape[2] == 1:
                img = img.squeeze()
            
            images.append(img)
            timestamps.append(timestamp)
            
            frame_count += 1
            
            if len(images) >= max_frames:
                break
    
    return images, timestamps


def load_ground_truth(rosbag_path, timestamps):
    """Load ground-truth poses from rosbag."""
    gt_poses = []
    
    with AnyReader([Path(rosbag_path)]) as reader:
        gt_data = {}
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == '/imu0':
                msg = reader.deserialize(rawdata, connection.msgtype)
                gt_data[timestamp] = msg
    
    for ts in timestamps:
        closest_ts = min(gt_data.keys(), key=lambda x: abs(x - ts))
        
        if abs(closest_ts - ts) < 1e8:
            msg = gt_data[closest_ts]
            
            position = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ])
            
            orientation = Rotation.from_quat([
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w
            ])
            
            pose = np.eye(4)
            pose[:3, :3] = orientation.as_matrix()
            pose[:3, 3] = position
            
            gt_poses.append(pose)
        else:
            gt_poses.append(np.eye(4))
    
    return gt_poses


def main():
    parser = argparse.ArgumentParser(description='Trajectory estimation')
    parser.add_argument('--calibration', type=str, default='calibration/stereo_calibration.pkl',
                        help='Calibration file')
    parser.add_argument('--rosbag', type=str, required=True, help='Path to rosbag directory')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--max_frames', type=int, default=200, help='Maximum number of frames')
    parser.add_argument('--skip_frames', type=int, default=5, help='Process every N frames')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading images from rosbag...")
    images, timestamps = load_images_from_rosbag(args.rosbag, args.max_frames, args.skip_frames)
    print(f"Loaded {len(images)} images")
    
    print("Loading ground-truth poses...")
    gt_poses = load_ground_truth(args.rosbag, timestamps)
    print(f"Loaded {len(gt_poses)} ground-truth poses")
    
    print("Initializing trajectory estimator...")
    estimator = TrajectoryEstimator(args.calibration)
    
    print("Estimating trajectory...")
    estimated_poses = estimator.estimate_trajectory(images, gt_poses)
    
    print("Visualizing trajectory...")
    estimator.visualize_trajectory(estimated_poses, gt_poses,
                                   output_dir / 'trajectory.png')
    
    if gt_poses:
        print("Computing trajectory errors...")
        errors = estimator.compute_trajectory_error(estimated_poses, gt_poses)
        
        print("\nTrajectory Error Metrics:")
        print(f"  ATE (Absolute Trajectory Error): {errors['ATE']:.4f} m")
        print(f"  RMSE: {errors['RMSE']:.4f} m")
        print(f"  Max Error: {errors['Max Error']:.4f} m")
        
        with open(output_dir / 'trajectory_errors.txt', 'w') as f:
            f.write("Trajectory Error Metrics\n")
            f.write("=" * 50 + "\n")
            f.write(f"ATE (Absolute Trajectory Error): {errors['ATE']:.4f} m\n")
            f.write(f"RMSE: {errors['RMSE']:.4f} m\n")
            f.write(f"Max Error: {errors['Max Error']:.4f} m\n")
    
    with open(output_dir / 'estimated_trajectory.pkl', 'wb') as f:
        pickle.dump(estimated_poses, f)
    
    print(f"\nResults saved to {output_dir}/")


if __name__ == '__main__':
    main()

