#!/bin/bash

ROSBAG_DIR="data/MH_01_easy"
ROSBAG_FILE="$ROSBAG_DIR/MH_01_easy_with_camera_info.db3"
EXPECTED_SIZE=2555000000

echo "Waiting for rosbag download to complete..."
echo "Expected size: ~2.5 GB"

while true; do
    if [ -f "$ROSBAG_FILE" ]; then
        CURRENT_SIZE=$(stat -f%z "$ROSBAG_FILE" 2>/dev/null || echo "0")
        PERCENT=$((CURRENT_SIZE * 100 / EXPECTED_SIZE))
        
        if [ $PERCENT -ge 95 ]; then
            echo "Download appears complete ($PERCENT%). Verifying..."
            if sqlite3 "$ROSBAG_FILE" "SELECT 1;" > /dev/null 2>&1; then
                echo "Rosbag file is valid!"
                break
            else
                echo "File may still be downloading or corrupted. Waiting..."
            fi
        else
            echo "Progress: $PERCENT% ($(ls -lh "$ROSBAG_FILE" | awk '{print $5}'))"
        fi
    fi
    sleep 30
done

echo ""
echo "Running all exercises..."
cd "$(dirname "$0")"
./run_all.sh "$ROSBAG_DIR" results

