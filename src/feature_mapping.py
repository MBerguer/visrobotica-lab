#!/usr/bin/env python3
"""
Feature and Dense Mapping with Ground-Truth Poses
Exercise 2f and 2i: Create 3D maps using ground-truth localization
"""

import cv2
import numpy as np
import pickle
import argparse
from pathlib import Path
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    from ground_truth_loader import load_ground_truth_for_images
except ImportError:
    load_ground_truth_for_images = None


class FeatureMapper:
    def __init__(self, calibration_file):
        with open(calibration_file, 'rb') as f:
            self.calib = pickle.load(f)
        
        self.prepare_rectification_maps()
        self.detector = cv2.ORB_create(nfeatures=2000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def prepare_rectification_maps(self):
        """Prepare rectification maps."""
        w, h = self.calib['image_size']
        
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            self.calib['K_left'], self.calib['dist_left'],
            self.calib.get('R1', np.eye(3)), self.calib.get('P1', self.calib['K_left']),
            (w, h), cv2.CV_32FC1
        )
        
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
            self.calib['K_right'], self.calib['dist_right'],
            self.calib.get('R2', np.eye(3)), self.calib.get('P2', self.calib['K_right']),
            (w, h), cv2.CV_32FC1
        )
    
    def rectify_images(self, img_left, img_right):
        """Rectify stereo images."""
        rect_left = cv2.remap(img_left, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        rect_right = cv2.remap(img_right, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
        return rect_left, rect_right
    
    def extract_and_match_features(self, img_left, img_right):
        """Extract and match features."""
        kp_left, desc_left = self.detector.detectAndCompute(img_left, None)
        kp_right, desc_right = self.detector.detectAndCompute(img_right, None)
        
        if desc_left is None or desc_right is None:
            return [], [], []
        
        matches = self.bf_matcher.match(desc_left, desc_right)
        matches = [m for m in matches if m.distance < 30]
        
        return kp_left, kp_right, matches
    
    def triangulate_points(self, kp_left, kp_right, matches):
        """Triangulate 3D points."""
        if len(matches) == 0:
            return np.array([]), np.array([])
        
        points_left = np.float32([kp_left[m.queryIdx].pt for m in matches])
        points_right = np.float32([kp_right[m.trainIdx].pt for m in matches])
        
        points_left = points_left.T.reshape(2, -1)
        points_right = points_right.T.reshape(2, -1)
        
        P1 = self.calib.get('P1', self.calib['K_left'] @ np.hstack([np.eye(3), np.zeros((3,1))]))
        P2 = self.calib.get('P2', self.calib['K_right'] @ np.hstack([self.calib['R'], self.calib['T']]))
        
        points_4d = cv2.triangulatePoints(P1, P2, points_left, points_right)
        points_3d = (points_4d[:3] / points_4d[3]).T
        
        return points_3d, points_left.T
    
    def compute_disparity(self, img_left, img_right):
        """Compute disparity map."""
        if len(img_left.shape) == 3:
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = img_left
            gray_right = img_right
        
        min_disp = 0
        num_disp = 16 * 10
        block_size = 5
        
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size**2,
            P2=32 * 3 * block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        return disparity
    
    def reconstruct_3d_from_disparity(self, disparity, img_left):
        """Dense 3D reconstruction."""
        Q = self.calib.get('Q')
        
        if Q is None:
            focal_length = self.calib['K_left'][0, 0]
            baseline = abs(self.calib['T'][0, 0])
            cx = self.calib['K_left'][0, 2]
            cy = self.calib['K_left'][1, 2]
            
            Q = np.float32([
                [1, 0, 0, -cx],
                [0, 1, 0, -cy],
                [0, 0, 0, focal_length],
                [0, 0, 1/baseline, 0]
            ])
        
        points_3d = cv2.reprojectImageTo3D(disparity, Q)
        
        mask = disparity > disparity.min()
        points_3d_filtered = points_3d[mask]
        
        if len(img_left.shape) == 3:
            colors = img_left[mask]
        else:
            colors = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)[mask]
        
        return points_3d_filtered, colors
    
    def transform_points(self, points_3d, pose):
        """Transform points to world frame."""
        points_homogeneous = np.hstack([points_3d, np.ones((len(points_3d), 1))])
        points_world = (pose @ points_homogeneous.T).T
        return points_world[:, :3]
    
    def create_feature_map(self, image_pairs, poses):
        """Create sparse feature map."""
        all_points = []
        all_colors = []
        
        for i, ((img_left, img_right), pose) in enumerate(zip(image_pairs, poses)):
            print(f"Processing frame {i+1}/{len(image_pairs)}...")
            
            rect_left, rect_right = self.rectify_images(img_left, img_right)
            kp_left, kp_right, matches = self.extract_and_match_features(rect_left, rect_right)
            
            if len(matches) == 0:
                continue
            
            points_3d, pts_img = self.triangulate_points(kp_left, kp_right, matches)
            
            mask = np.isfinite(points_3d).all(axis=1)
            mask &= (np.abs(points_3d[:, 2]) < 100)
            points_3d = points_3d[mask]
            pts_img = pts_img[mask]
            
            if len(points_3d) == 0:
                continue
            
            points_world = self.transform_points(points_3d, pose)
            
            colors = rect_left[pts_img[:, 1].astype(int), pts_img[:, 0].astype(int)]
            if len(colors.shape) == 1:
                colors = np.stack([colors, colors, colors], axis=-1)
            
            all_points.append(points_world)
            all_colors.append(colors)
        
        if all_points:
            all_points = np.vstack(all_points)
            all_colors = np.vstack(all_colors)
        else:
            all_points = np.array([])
            all_colors = np.array([])
        
        return all_points, all_colors
    
    def create_dense_map(self, image_pairs, poses, subsample=4):
        """Create dense map from disparity."""
        all_points = []
        all_colors = []
        
        for i, ((img_left, img_right), pose) in enumerate(zip(image_pairs, poses)):
            print(f"Processing frame {i+1}/{len(image_pairs)} for dense mapping...")
            
            rect_left, rect_right = self.rectify_images(img_left, img_right)
            
            disparity = self.compute_disparity(rect_left, rect_right)
            
            points_3d, colors = self.reconstruct_3d_from_disparity(disparity, rect_left)
            
            if len(points_3d) == 0:
                continue
            
            if subsample > 1:
                indices = np.random.choice(len(points_3d), len(points_3d)//subsample, replace=False)
                points_3d = points_3d[indices]
                colors = colors[indices]
            
            points_world = self.transform_points(points_3d, pose)
            
            all_points.append(points_world)
            all_colors.append(colors)
        
        if all_points:
            all_points = np.vstack(all_points)
            all_colors = np.vstack(all_colors)
        else:
            all_points = np.array([])
            all_colors = np.array([])
        
        return all_points, all_colors
    
    def save_point_cloud(self, points_3d, colors, output_path):
        """Save point cloud in PLY format."""
        points_3d = points_3d.reshape(-1, 3)
        colors = colors.reshape(-1, 3)
        
        mask = np.isfinite(points_3d).all(axis=1)
        mask &= (np.abs(points_3d).max(axis=1) < 1000)
        
        points_3d = points_3d[mask]
        colors = colors[mask]
        
        with open(output_path, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(points_3d)}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')
            
            for pt, col in zip(points_3d, colors):
                f.write(f'{pt[0]} {pt[1]} {pt[2]} {int(col[2])} {int(col[1])} {int(col[0])}\n')


def load_data_from_rosbag(rosbag_path, max_frames=50, skip_frames=10, gt_csv=None):
    """Load stereo images and poses from rosbag.
    
    Args:
        rosbag_path: Path to rosbag directory
        max_frames: Maximum number of frames to process
        skip_frames: Process every N frames
        gt_csv: Optional path to ground-truth CSV file
    """
    image_pairs = []
    poses = []
    timestamps = []
    
    with AnyReader([Path(rosbag_path)]) as reader:
        left_msgs = {}
        right_msgs = {}
        
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == '/cam0/image_raw':
                msg = reader.deserialize(rawdata, connection.msgtype)
                left_msgs[timestamp] = msg
            elif connection.topic == '/cam1/image_raw':
                msg = reader.deserialize(rawdata, connection.msgtype)
                right_msgs[timestamp] = msg
    
    left_times = sorted(left_msgs.keys())
    
    frame_count = 0
    for i, left_t in enumerate(left_times):
        if i % skip_frames != 0:
            continue
        
        right_times = sorted(right_msgs.keys())
        closest_right_t = min(right_times, key=lambda x: abs(x - left_t))
        
        if abs(closest_right_t - left_t) > 1e8:
            continue
        
        left_msg = left_msgs[left_t]
        right_msg = right_msgs[closest_right_t]
        
        left_img = np.frombuffer(left_msg.data, dtype=np.uint8).reshape(
            left_msg.height, left_msg.width, -1
        )
        right_img = np.frombuffer(right_msg.data, dtype=np.uint8).reshape(
            right_msg.height, right_msg.width, -1
        )
        
        if left_img.shape[2] == 1:
            left_img = left_img.squeeze()
        if right_img.shape[2] == 1:
            right_img = right_img.squeeze()
        
        image_pairs.append((left_img, right_img))
        timestamps.append(left_t)
        
        frame_count += 1
        if frame_count >= max_frames:
            break
    
    if gt_csv and Path(gt_csv).exists():
        if load_ground_truth_for_images is None:
            print("Warning: ground_truth_loader not available, using synthetic poses")
            poses = []
            for i in range(len(image_pairs)):
                pose = np.eye(4)
                pose[:3, 3] = [i * 0.1, 0, 0]
                poses.append(pose)
        else:
            print(f"Loading ground-truth from {gt_csv}...")
            poses = load_ground_truth_for_images(gt_csv, timestamps)
            print(f"Loaded {len(poses)} ground-truth poses")
    else:
        print("Using synthetic poses (no ground-truth CSV provided)")
        poses = []
        for i in range(len(image_pairs)):
            pose = np.eye(4)
            pose[:3, 3] = [i * 0.1, 0, 0]
            poses.append(pose)
    
    return image_pairs, poses


def main():
    parser = argparse.ArgumentParser(description='Feature and dense mapping')
    parser.add_argument('--calibration', type=str, default='calibration/stereo_calibration.pkl',
                        help='Calibration file')
    parser.add_argument('--rosbag', type=str, required=True, help='Path to rosbag directory')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--max_frames', type=int, default=50, help='Maximum number of frames')
    parser.add_argument('--skip_frames', type=int, default=10, help='Process every N frames')
    parser.add_argument('--mode', type=str, choices=['sparse', 'dense', 'both'], default='both',
                        help='Mapping mode')
    parser.add_argument('--gt_csv', type=str, default=None,
                        help='Optional path to ground-truth CSV file')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data from rosbag...")
    image_pairs, poses = load_data_from_rosbag(args.rosbag, args.max_frames, args.skip_frames, args.gt_csv)
    print(f"Loaded {len(image_pairs)} image pairs and {len(poses)} poses")
    
    print("Initializing mapper...")
    mapper = FeatureMapper(args.calibration)
    
    if args.mode in ['sparse', 'both']:
        print("Creating sparse feature map...")
        points_sparse, colors_sparse = mapper.create_feature_map(image_pairs, poses)
        
        if len(points_sparse) > 0:
            mapper.save_point_cloud(points_sparse, colors_sparse,
                                   output_dir / 'map_sparse_gt.ply')
            print(f"Sparse map saved with {len(points_sparse)} points")
    
    if args.mode in ['dense', 'both']:
        print("Creating dense map...")
        points_dense, colors_dense = mapper.create_dense_map(image_pairs, poses, subsample=8)
        
        if len(points_dense) > 0:
            mapper.save_point_cloud(points_dense, colors_dense,
                                   output_dir / 'map_dense_gt.ply')
            print(f"Dense map saved with {len(points_dense)} points")
    
    print(f"\nResults saved to {output_dir}/")


if __name__ == '__main__':
    main()

