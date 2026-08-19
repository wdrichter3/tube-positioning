import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from .contour_detector import ContourTubeDetector
from .hough_detector import HoughTubeDetector
from .hybrid_detector import HybridTubeDetector
from .overlay import draw_detection_overlay

def _parse_roi(text, shape):
    if not text:
        return None
    vals=[int(v.strip()) for v in text.split(",")]
    if len(vals)!=4:
        raise ValueError("ROI must be x,y,w,h")
    x,y,w,h=vals
    mask=np.zeros(shape[:2],np.uint8)
    mask[max(0,y):min(shape[0],y+h), max(0,x):min(shape[1],x+w)]=255
    return mask

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--detector", choices=["contour","hough","hybrid"], default="hybrid")
    ap.add_argument("--roi", help="x,y,w,h")
    ap.add_argument("--min-diameter", type=float, default=12)
    ap.add_argument("--max-diameter", type=float, default=100)
    ap.add_argument("--overlay", default="tube_detection_overlay.jpg")
    ap.add_argument("--json", dest="json_path", default="tube_candidates.json")
    args=ap.parse_args()
    image=cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not load image: {args.image}")
    roi=_parse_roi(args.roi,image.shape)
    kwargs=dict(min_diameter_px=args.min_diameter,max_diameter_px=args.max_diameter)
    if args.detector=="contour":
        det=ContourTubeDetector(**kwargs)
    elif args.detector=="hough":
        det=HoughTubeDetector(**kwargs)
    else:
        det=HybridTubeDetector(ContourTubeDetector(**kwargs),HoughTubeDetector(**kwargs))
    result=det.detect(image,roi)
    overlay=draw_detection_overlay(image,result)
    cv2.imwrite(args.overlay,overlay)
    data=[c.__dict__ for c in result.candidates]
    Path(args.json_path).write_text(json.dumps(data,indent=2))
    print(f"detector={args.detector} candidates={len(data)} overlay={args.overlay} json={args.json_path}")

if __name__=="__main__":
    main()
