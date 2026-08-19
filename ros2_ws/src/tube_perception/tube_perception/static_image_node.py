import json
import os
from pathlib import Path
import cv2
import rclpy
from rclpy.node import Node
from tube_interfaces.msg import TubeCandidate

# Import the shared application library from the project source tree.
from . import HybridTubeDetector, draw_detection_overlay

class StaticImageNode(Node):
    def __init__(self):
        super().__init__("static_image_tube_detector")
        self.declare_parameter("image_path", "")
        self.declare_parameter("overlay_path", "/tmp/tube_detection_overlay.jpg")
        self.declare_parameter("min_diameter_px", 12.0)
        self.declare_parameter("max_diameter_px", 100.0)
        self.publisher = self.create_publisher(TubeCandidate, "/perception/candidates", 100)
        self.timer = self.create_timer(0.25, self.process_once)
        self.done = False

    def process_once(self):
        if self.done:
            return
        self.done = True
        image_path = self.get_parameter("image_path").value
        if not image_path:
            self.get_logger().error("image_path parameter is required")
            return
        image = cv2.imread(image_path)
        if image is None:
            self.get_logger().error(f"Could not load image: {image_path}")
            return
        det = HybridTubeDetector()
        result = det.detect(image)
        for c in result.candidates:
            msg = TubeCandidate()
            msg.candidate_id = int(c.candidate_id)
            msg.pixel_x = float(c.pixel_x)
            msg.pixel_y = float(c.pixel_y)
            msg.major_axis_px = float(c.major_axis_px)
            msg.minor_axis_px = float(c.minor_axis_px)
            msg.orientation_deg = float(c.orientation_deg)
            msg.estimated_diameter_mm = float(c.estimated_diameter_mm)
            msg.visual_confidence = float(c.visual_confidence)
            msg.detector_name = c.detector_name
            self.publisher.publish(msg)
        overlay = draw_detection_overlay(image, result)
        overlay_path = self.get_parameter("overlay_path").value
        cv2.imwrite(overlay_path, overlay)
        self.get_logger().info(
            f"Detected {len(result.candidates)} candidates; overlay saved to {overlay_path}"
        )

def main(args=None):
    rclpy.init(args=args)
    node=StaticImageNode()
    rclpy.spin_once(node, timeout_sec=1.0)
    node.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()
