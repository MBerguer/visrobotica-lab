#!/usr/bin/env python3
"""
Stereo Vision Pipeline
Implements all exercises: rectification, feature extraction, matching, triangulation, 
RANSAC filtering, disparity map, and dense reconstruction.
"""

import cv2
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from rosbags.highlevel import AnyReader


class StereoPipeline:
    def __init__(self, calibration_file):
        with open(calibration_file, 'rb') as f:
            self.calib = pickle.load(f)
        
        self.prepare_rectification_maps()
    
    def prepare_rectification_maps(self):
        """Prepare rectification maps for faster processing."""
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
        """Exercise 2a: Rectify stereo images."""
        rect_left = cv2.remap(img_left, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        rect_right = cv2.remap(img_right, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
        return rect_left, rect_right
    
    def extract_features(self, image, detector_type='ORB'):
        """Exercise 2b: Extract features (keypoints and descriptors)."""
        if detector_type == 'ORB':
            detector = cv2.ORB_create(nfeatures=2000)
        elif detector_type == 'SIFT':
            detector = cv2.SIFT_create(nfeatures=2000)
        elif detector_type == 'FAST':
            detector = cv2.FastFeatureDetector_create()
            descriptor = cv2.ORB_create()
            keypoints = detector.detect(image, None)
            keypoints, descriptors = descriptor.compute(image, keypoints)
            return keypoints, descriptors
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
        
        keypoints, descriptors = detector.detectAndCompute(image, None)
        return keypoints, descriptors
    
    def match_features(self, desc_left, desc_right, max_distance=None):
        """Exercise 2c: Match features between left and right images."""
        if desc_left.dtype == np.uint8:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        matches = bf.match(desc_left, desc_right)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if max_distance is not None:
            matches = [m for m in matches if m.distance < max_distance]
        
        return matches
    
    def triangulate_points(self, kp_left, kp_right, matches):
        """Exercise 2d: Triangulate 3D points from matches."""
        points_left = np.float32([kp_left[m.queryIdx].pt for m in matches])
        points_right = np.float32([kp_right[m.trainIdx].pt for m in matches])
        
        points_left = points_left.T.reshape(2, -1)
        points_right = points_right.T.reshape(2, -1)
        
        P1 = self.calib.get('P1', self.calib['K_left'] @ np.hstack([np.eye(3), np.zeros((3,1))]))
        P2 = self.calib.get('P2', self.calib['K_right'] @ np.hstack([self.calib['R'], self.calib['T']]))
        
        points_4d = cv2.triangulatePoints(P1.astype(np.float64), P2.astype(np.float64), 
                                         points_left, points_right)
        points_3d = (points_4d[:3] / points_4d[3]).T
        
        valid_mask = (
            np.isfinite(points_3d).all(axis=1) &
            (np.abs(points_3d[:, 2]) > 0.1) &
            (np.abs(points_3d[:, 2]) < 100.0)
        )
        
        points_3d = points_3d[valid_mask]
        points_left_valid = points_left.T[valid_mask]
        points_right_valid = points_right.T[valid_mask]
        
        return points_3d, points_left_valid, points_right_valid
    
    def filter_matches_ransac(self, kp_left, kp_right, matches, threshold=5.0):
        """Exercise 2e: Filter spurious matches using RANSAC."""
        if len(matches) < 4:
            return [], None
        
        points_left = np.float32([kp_left[m.queryIdx].pt for m in matches])
        points_right = np.float32([kp_right[m.trainIdx].pt for m in matches])
        
        H, mask = cv2.findHomography(points_left, points_right, cv2.RANSAC, threshold)
        
        inlier_matches = [m for m, inlier in zip(matches, mask.ravel()) if inlier]
        
        return inlier_matches, H
    
    def compute_disparity_map(self, rect_left, rect_right, method='SGBM'):
        """Exercise 2g: Compute disparity map."""
        if len(rect_left.shape) == 3:
            gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = rect_left
            gray_right = rect_right
        
        if method == 'SGBM':
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
        else:
            stereo = cv2.StereoBM_create(numDisparities=16*10, blockSize=15)
        
        disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        
        invalid_mask = (disparity <= 0) | (disparity == np.inf) | np.isnan(disparity)
        disparity[invalid_mask] = 0
        
        return disparity
    
    def reconstruct_3d_dense(self, disparity, img_left):
        """Exercise 2h: Dense 3D reconstruction from disparity map."""
        Q = self.calib.get('Q')
        
        if Q is None:
            focal_length = self.calib['K_left'][0, 0]
            baseline = abs(self.calib['T'][0, 0])
            cx = self.calib['K_left'][0, 2]
            cy = self.calib['K_left'][1, 2]
            
            Q = np.float64([
                [1, 0, 0, -cx],
                [0, 1, 0, -cy],
                [0, 0, 0, focal_length],
                [0, 0, 1/baseline, 0]
            ])
        else:
            Q = Q.astype(np.float64)
        
        points_3d = cv2.reprojectImageTo3D(disparity.astype(np.float32), Q.astype(np.float32))
        points_3d = points_3d.astype(np.float64)
        
        valid_mask = (
            (disparity > disparity.min()) &
            np.isfinite(points_3d).all(axis=2) &
            (np.abs(points_3d[:, :, 2]) > 0.1) &
            (np.abs(points_3d[:, :, 2]) < 100.0) &
            (np.abs(points_3d[:, :, 0]) < 1000.0) &
            (np.abs(points_3d[:, :, 1]) < 1000.0)
        )
        
        points_3d_filtered = points_3d[valid_mask]
        
        if len(img_left.shape) == 3:
            colors = img_left[valid_mask]
        else:
            colors = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)[valid_mask]
        
        return points_3d_filtered, colors
    
    def estimate_pose_monocular(self, kp1, kp2, matches):
        """Exercise 2j: Estimate pose using monocular vision."""
        points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        points2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        E, mask = cv2.findEssentialMat(
            points1, points2,
            self.calib['K_left'],
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        
        _, R, t, mask_pose = cv2.recoverPose(E, points1, points2, self.calib['K_left'], mask=mask)
        
        baseline = np.linalg.norm(self.calib['T'])
        t_scaled = t * baseline
        
        return R, t_scaled, mask_pose
    
    def visualize_features(self, img_left, img_right, kp_left, kp_right, output_path):
        """Visualize extracted features."""
        img_left_kp = cv2.drawKeypoints(img_left, kp_left, None, color=(0, 255, 0))
        img_right_kp = cv2.drawKeypoints(img_right, kp_right, None, color=(0, 255, 0))
        
        combined = np.hstack([img_left_kp, img_right_kp])
        
        plt.figure(figsize=(15, 5))
        plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        plt.title(f'Features: {len(kp_left)} left, {len(kp_right)} right')
        plt.axis('off')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_matches(self, img_left, img_right, kp_left, kp_right, matches, output_path, title=''):
        """Visualize feature matches."""
        img_matches = cv2.drawMatches(
            img_left, kp_left, img_right, kp_right, matches[:100],
            None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        plt.figure(figsize=(20, 10))
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title(f'{title} - {len(matches)} matches (showing first 100)')
        plt.axis('off')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_disparity(self, disparity, output_path):
        """Visualize disparity map."""
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB))
        plt.title('Disparity Map')
        plt.colorbar(label='Disparity')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_point_cloud(self, points_3d, colors, output_path):
        """Save point cloud in PLY format."""
        points_3d = points_3d.reshape(-1, 3)
        colors = colors.reshape(-1, 3)
        
        mask = np.isfinite(points_3d).all(axis=1)
        mask &= (np.abs(points_3d[:, 2]) < 100)
        
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
                f.write(f'{pt[0]} {pt[1]} {pt[2]} {col[2]} {col[1]} {col[0]}\n')


def load_images_from_rosbag(rosbag_path, frame_idx=0):
    """Load a stereo pair from rosbag."""
    with AnyReader([Path(rosbag_path)]) as reader:
        left_msgs = []
        right_msgs = []
        
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == '/cam0/image_raw':
                msg = reader.deserialize(rawdata, connection.msgtype)
                left_msgs.append((timestamp, msg))
            elif connection.topic == '/cam1/image_raw':
                msg = reader.deserialize(rawdata, connection.msgtype)
                right_msgs.append((timestamp, msg))
        
        if frame_idx >= len(left_msgs) or frame_idx >= len(right_msgs):
            frame_idx = min(len(left_msgs), len(right_msgs)) // 2
        
        left_msg = left_msgs[frame_idx][1]
        right_msg = right_msgs[frame_idx][1]
        
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
        
        return left_img, right_img


def main():
    parser = argparse.ArgumentParser(description='Stereo vision pipeline')
    parser.add_argument('--calibration', type=str, default='calibration/stereo_calibration.pkl',
                        help='Calibration file')
    parser.add_argument('--rosbag', type=str, help='Path to rosbag directory')
    parser.add_argument('--frame', type=int, default=100, help='Frame index to process')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--detector', type=str, default='ORB', choices=['ORB', 'SIFT', 'FAST'],
                        help='Feature detector type')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Initializing stereo pipeline...")
    pipeline = StereoPipeline(args.calibration)
    
    print(f"Loading images from rosbag (frame {args.frame})...")
    if args.rosbag:
        img_left, img_right = load_images_from_rosbag(args.rosbag, args.frame)
    else:
        print("Error: --rosbag argument required")
        return
    
    print("Exercise 2a: Rectifying images...")
    rect_left, rect_right = pipeline.rectify_images(img_left, img_right)
    cv2.imwrite(str(output_dir / 'rectified_left.png'), rect_left)
    cv2.imwrite(str(output_dir / 'rectified_right.png'), rect_right)
    
    print(f"Exercise 2b: Extracting features ({args.detector})...")
    kp_left, desc_left = pipeline.extract_features(rect_left, args.detector)
    kp_right, desc_right = pipeline.extract_features(rect_right, args.detector)
    pipeline.visualize_features(rect_left, rect_right, kp_left, kp_right,
                                output_dir / 'features.png')
    
    print("Exercise 2c: Matching features...")
    matches_all = pipeline.match_features(desc_left, desc_right)
    pipeline.visualize_matches(rect_left, rect_right, kp_left, kp_right, matches_all,
                              output_dir / 'matches_all.png', 'All Matches')
    
    matches_filtered = pipeline.match_features(desc_left, desc_right, max_distance=30)
    pipeline.visualize_matches(rect_left, rect_right, kp_left, kp_right, matches_filtered,
                              output_dir / 'matches_filtered.png', 'Matches (distance < 30)')
    
    print("Exercise 2d: Triangulating 3D points...")
    points_3d, pts_left, pts_right = pipeline.triangulate_points(kp_left, kp_right, matches_filtered)
    colors = rect_left[pts_left[:, 1].astype(int), pts_left[:, 0].astype(int)]
    if len(colors.shape) == 1:
        colors = np.stack([colors, colors, colors], axis=-1)
    pipeline.save_point_cloud(points_3d, colors, output_dir / 'pointcloud_sparse.ply')
    
    print("Exercise 2e: Filtering with RANSAC...")
    inlier_matches, H = pipeline.filter_matches_ransac(kp_left, kp_right, matches_filtered)
    pipeline.visualize_matches(rect_left, rect_right, kp_left, kp_right, inlier_matches,
                              output_dir / 'matches_ransac.png', 'RANSAC Inliers')
    
    if H is not None:
        points_3d_ransac, pts_left_r, pts_right_r = pipeline.triangulate_points(
            kp_left, kp_right, inlier_matches
        )
        colors_ransac = rect_left[pts_left_r[:, 1].astype(int), pts_left_r[:, 0].astype(int)]
        if len(colors_ransac.shape) == 1:
            colors_ransac = np.stack([colors_ransac, colors_ransac, colors_ransac], axis=-1)
        pipeline.save_point_cloud(points_3d_ransac, colors_ransac,
                                 output_dir / 'pointcloud_sparse_ransac.ply')
    
    print("Exercise 2g: Computing disparity map...")
    disparity = pipeline.compute_disparity_map(rect_left, rect_right)
    pipeline.visualize_disparity(disparity, output_dir / 'disparity_map.png')
    np.save(output_dir / 'disparity.npy', disparity)
    
    print("Exercise 2h: Dense 3D reconstruction...")
    points_3d_dense, colors_dense = pipeline.reconstruct_3d_dense(disparity, rect_left)
    pipeline.save_point_cloud(points_3d_dense, colors_dense,
                             output_dir / 'pointcloud_dense.ply')
    
    print(f"\nResults saved to {output_dir}/")
    print(f"  - Rectified images")
    print(f"  - Feature visualizations")
    print(f"  - Match visualizations (all, filtered, RANSAC)")
    print(f"  - Sparse point clouds (.ply)")
    print(f"  - Disparity map")
    print(f"  - Dense point cloud (.ply)")
    print("\nUse MeshLab or CloudCompare to visualize .ply files")


if __name__ == '__main__':
    main()

