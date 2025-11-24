#!/bin/bash

set -e

ROSBAG_PATH="${1:-data/euroc_rosbag}"
OUTPUT_DIR="${2:-results}"

echo "========================================"
echo "  Stereo Vision Pipeline - Full Run"
echo "========================================"
echo "Rosbag: $ROSBAG_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p calibration
mkdir -p images

echo "[1/5] Running camera calibration..."
python3 src/camera_calibration.py \
    --output calibration/stereo_calibration.pkl

echo ""
echo "[2/5] Running stereo pipeline (all exercises 2a-2h)..."
python3 src/stereo_pipeline.py \
    --calibration calibration/stereo_calibration.pkl \
    --rosbag "$ROSBAG_PATH" \
    --frame 100 \
    --output_dir "$OUTPUT_DIR" \
    --detector ORB

echo ""
echo "[3/5] Running trajectory estimation (exercise 2j)..."
python3 src/trajectory_estimation.py \
    --calibration calibration/stereo_calibration.pkl \
    --rosbag "$ROSBAG_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_frames 200 \
    --skip_frames 5

echo ""
echo "[4/5] Running feature mapping (optional exercise 2f)..."
python3 src/feature_mapping.py \
    --calibration calibration/stereo_calibration.pkl \
    --rosbag "$ROSBAG_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_frames 50 \
    --skip_frames 10 \
    --mode sparse

echo ""
echo "[5/5] Running dense mapping (optional exercise 2i)..."
python3 src/feature_mapping.py \
    --calibration calibration/stereo_calibration.pkl \
    --rosbag "$ROSBAG_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_frames 30 \
    --skip_frames 15 \
    --mode dense

echo ""
echo "========================================"
echo "  All exercises completed!"
echo "========================================"
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Generated files:"
echo "  - Calibration data: calibration/stereo_calibration.pkl"
echo "  - Rectified images: $OUTPUT_DIR/rectified_*.png"
echo "  - Feature visualizations: $OUTPUT_DIR/features.png"
echo "  - Match visualizations: $OUTPUT_DIR/matches_*.png"
echo "  - Sparse point clouds: $OUTPUT_DIR/pointcloud_sparse*.ply"
echo "  - Disparity map: $OUTPUT_DIR/disparity_map.png"
echo "  - Dense point cloud: $OUTPUT_DIR/pointcloud_dense.ply"
echo "  - Trajectory: $OUTPUT_DIR/trajectory.png"
echo "  - Feature map: $OUTPUT_DIR/map_sparse_gt.ply"
echo "  - Dense map: $OUTPUT_DIR/map_dense_gt.ply"
echo ""
echo "View .ply files with MeshLab or CloudCompare"


