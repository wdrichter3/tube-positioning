# Perception V0.1

This increment adds the first reusable tube-candidate detection layer.

## Algorithms

- `ContourTubeDetector`: adaptive threshold + contour/ellipse filtering.
- `HoughTubeDetector`: Hough-circle baseline.
- `HybridTubeDetector`: confidence-weighted merging of contour and Hough observations.

The hybrid result is still **candidate detection only**. It does not assign tube IDs, fit pitch, infer missing tubes, or remove candidates by lattice geometry. Those belong in the next `tube_geometry` increment.

## Static CLI

From the project root with the Python venv active:

```bash
ros2 run tube_perception detect_image IMAGE_PATH \
  --detector hybrid \
  --min-diameter 20 --max-diameter 60 \
  --roi x,y,w,h \
  --overlay /tmp/tubes.jpg
```

## ROS node

The ROS node is intentionally thin. The detector algorithms and ROS wrapper live together in the `tube_perception` ament-python package for V0.1 so ROS can run them without mixing the project virtual environment into the ROS build.

Build ROS with the Python venv deactivated:

```bash
cd ~/projects/tube-positioning/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Run:

```bash
ros2 run tube_perception static_image_node --ros-args \
  -p image_path:=/absolute/path/to/image.jpg \
  -p overlay_path:=/tmp/tube_overlay.jpg
```

The node publishes each `tube_interfaces/msg/TubeCandidate` to `/perception/candidates` and writes a debug overlay.
