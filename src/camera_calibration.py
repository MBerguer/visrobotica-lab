#!/usr/bin/env python3
"""
Camera Calibration Script
Performs stereo camera calibration using checkerboard patterns from rosbag data.
"""

import cv2
import numpy as np
import pickle
from pathlib import Path
from rosbags.highlevel import AnyReader
import argparse


class StereoCameraCalibrator:
    def __init__(self, checkerboard_size=(9, 6), square_size=0.05):
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        self.objpoints = []
        self.imgpoints_left = []
        self.imgpoints_right = []
        
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
    def find_chessboard_corners(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
        
        if ret:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            return True, corners_refined
        return False, None
    
    def add_calibration_data(self, img_left, img_right):
        ret_left, corners_left = self.find_chessboard_corners(img_left)
        ret_right, corners_right = self.find_chessboard_corners(img_right)
        
        if ret_left and ret_right:
            self.objpoints.append(self.objp)
            self.imgpoints_left.append(corners_left)
            self.imgpoints_right.append(corners_right)
            return True
        return False
    
    def calibrate_stereo(self, image_size):
        print(f"Calibrating with {len(self.objpoints)} image pairs...")
        
        flags = cv2.CALIB_FIX_INTRINSIC
        ret_left, K_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_left, image_size, None, None
        )
        
        ret_right, K_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_right, image_size, None, None
        )
        
        print(f"Left camera RMS error: {ret_left:.4f}")
        print(f"Right camera RMS error: {ret_right:.4f}")
        
        flags = cv2.CALIB_FIX_INTRINSIC
        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        
        ret_stereo, K_left, dist_left, K_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_left, self.imgpoints_right,
            K_left, dist_left, K_right, dist_right,
            image_size, criteria=stereocalib_criteria, flags=flags
        )
        
        print(f"Stereo calibration RMS error: {ret_stereo:.4f}")
        
        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
            K_left, dist_left, K_right, dist_right,
            image_size, R, T, alpha=0
        )
        
        calibration_data = {
            'K_left': K_left,
            'dist_left': dist_left,
            'K_right': K_right,
            'dist_right': dist_right,
            'R': R,
            'T': T,
            'E': E,
            'F': F,
            'R1': R1,
            'R2': R2,
            'P1': P1,
            'P2': P2,
            'Q': Q,
            'roi_left': roi_left,
            'roi_right': roi_right,
            'image_size': image_size,
            'rms_error': ret_stereo
        }
        
        return calibration_data


def extract_calibration_images_from_rosbag(rosbag_path, output_dir, max_images=50):
    """Extract stereo images from rosbag for calibration."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    left_images = []
    right_images = []
    
    print(f"Reading rosbag from {rosbag_path}...")
    
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
    
    print(f"Found {len(left_msgs)} left images and {len(right_msgs)} right images")
    
    left_times = sorted(left_msgs.keys())
    right_times = sorted(right_msgs.keys())
    
    count = 0
    for left_t in left_times:
        closest_right_t = min(right_times, key=lambda x: abs(x - left_t))
        
        if abs(closest_right_t - left_t) < 1e8:
            left_msg = left_msgs[left_t]
            right_msg = right_msgs[closest_right_t]
            
            left_img = np.frombuffer(left_msg.data, dtype=np.uint8).reshape(
                left_msg.height, left_msg.width, -1
            )
            right_img = np.frombuffer(right_msg.data, dtype=np.uint8).reshape(
                right_msg.height, right_msg.width, -1
            )
            
            left_images.append(left_img)
            right_images.append(right_img)
            
            count += 1
            if count >= max_images:
                break
    
    return left_images, right_images


def main():
    parser = argparse.ArgumentParser(description='Stereo camera calibration')
    parser.add_argument('--rosbag', type=str, help='Path to rosbag directory')
    parser.add_argument('--output', type=str, default='calibration/stereo_calibration.pkl',
                        help='Output calibration file')
    parser.add_argument('--checkerboard_rows', type=int, default=9,
                        help='Number of internal corners in rows')
    parser.add_argument('--checkerboard_cols', type=int, default=6,
                        help='Number of internal corners in columns')
    parser.add_argument('--square_size', type=float, default=0.05,
                        help='Checkerboard square size in meters')
    
    args = parser.parse_args()
    
    if args.rosbag:
        left_images, right_images = extract_calibration_images_from_rosbag(
            args.rosbag, 'calibration/images'
        )
    else:
        print("Note: Using simulated calibration from EuRoC dataset parameters")
        print("For actual calibration, provide --rosbag with checkerboard images")
        
        calib_data = {
            'K_left': np.array([[458.654, 0, 367.215],
                                [0, 457.296, 248.375],
                                [0, 0, 1]]),
            'dist_left': np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0]),
            'K_right': np.array([[457.587, 0, 379.999],
                                 [0, 456.134, 255.238],
                                 [0, 0, 1]]),
            'dist_right': np.array([-0.28368365, 0.07451284, -0.00010473, -3.55590700e-05, 0.0]),
            'R': np.eye(3),
            'T': np.array([[-0.110074], [0], [0]]),
            'image_size': (752, 480)
        }
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(args.output, 'wb') as f:
            pickle.dump(calib_data, f)
        
        print(f"Calibration data saved to {args.output}")
        return
    
    if not left_images or not right_images:
        print("Error: Could not load images")
        return
    
    calibrator = StereoCameraCalibrator(
        checkerboard_size=(args.checkerboard_rows, args.checkerboard_cols),
        square_size=args.square_size
    )
    
    print(f"Processing {len(left_images)} image pairs...")
    valid_pairs = 0
    
    for i, (left_img, right_img) in enumerate(zip(left_images, right_images)):
        if calibrator.add_calibration_data(left_img, right_img):
            valid_pairs += 1
            print(f"Valid pair {valid_pairs} found (image {i+1})")
    
    print(f"Found {valid_pairs} valid calibration pairs")
    
    if valid_pairs < 10:
        print("Error: Not enough valid calibration pairs found")
        return
    
    h, w = left_images[0].shape[:2]
    calibration_data = calibrator.calibrate_stereo((w, h))
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, 'wb') as f:
        pickle.dump(calibration_data, f)
    
    print(f"Calibration data saved to {args.output}")
    print("\nCalibration Results:")
    print(f"Left Camera Matrix:\n{calibration_data['K_left']}")
    print(f"Right Camera Matrix:\n{calibration_data['K_right']}")
    print(f"Rotation:\n{calibration_data['R']}")
    print(f"Translation:\n{calibration_data['T']}")
    print(f"RMS Error: {calibration_data['rms_error']:.4f}")


if __name__ == '__main__':
    main()

