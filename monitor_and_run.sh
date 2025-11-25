#!/bin/bash
ROSBAG_FILE="data/MH_01_easy/MH_01_easy_with_camera_info.db3"
EXPECTED_SIZE=2555000000

echo "Monitoring download progress..."
while true; do
    if [ -f "$ROSBAG_FILE" ]; then
        size=$(stat -f%z "$ROSBAG_FILE" 2>/dev/null || echo "0")
        percent=$((size * 100 / EXPECTED_SIZE))
        current_size=$(ls -lh "$ROSBAG_FILE" | awk '{print $5}')
        
        echo "[$(date '+%H:%M:%S')] Progress: $percent% ($current_size)"
        
        if [ $percent -ge 98 ]; then
            echo "Checking if file is valid..."
            if sqlite3 "$ROSBAG_FILE" "SELECT 1;" > /dev/null 2>&1; then
                echo "✓ Download complete and file is valid!"
                echo "Starting all exercises..."
                ./run_all.sh data/MH_01_easy results
                exit 0
            fi
        fi
    fi
    sleep 60
done
