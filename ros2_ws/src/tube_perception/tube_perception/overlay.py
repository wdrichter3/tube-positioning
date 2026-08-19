import cv2
import numpy as np

def draw_detection_overlay(image, result, draw_rejected=False):
    out = image.copy()
    for c in result.candidates:
        center = (int(round(c.pixel_x)), int(round(c.pixel_y)))
        axes = (max(1, int(round(c.major_axis_px/2))), max(1, int(round(c.minor_axis_px/2))))
        cv2.ellipse(out, center, axes, c.orientation_deg, 0, 360, (0, 255, 0), 2)
        cv2.circle(out, center, 2, (0, 0, 255), -1)
        cv2.putText(out, str(c.candidate_id), (center[0]+3, center[1]-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    if draw_rejected:
        for x, y, _reason in result.rejected:
            p=(int(round(x)),int(round(y)))
            cv2.drawMarker(out,p,(0,165,255),cv2.MARKER_TILTED_CROSS,8,1)
    return out
