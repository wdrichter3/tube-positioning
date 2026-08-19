import cv2
import numpy as np
from tube_perception import ContourTubeDetector, HybridTubeDetector

def synthetic_sheet(rows=5, cols=7, pitch=45, radius=12):
    h=rows*pitch+80
    w=cols*pitch+80
    image=np.full((h,w,3),220,np.uint8)
    for r in range(rows):
        for c in range(cols):
            x=40+c*pitch
            y=40+r*pitch
            cv2.circle(image,(x,y),radius,(20,20,20),-1)
    return image, rows*cols

def test_contour_detector_finds_synthetic_lattice():
    image,n=synthetic_sheet()
    d=ContourTubeDetector(min_diameter_px=15,max_diameter_px=35,min_circularity=0.5)
    result=d.detect(image)
    assert len(result.candidates) >= int(n*0.9)

def test_hybrid_detector_no_gross_duplicates():
    image,n=synthetic_sheet()
    d=HybridTubeDetector()
    result=d.detect(image)
    assert len(result.candidates) >= int(n*0.8)
    assert len(result.candidates) <= int(n*1.25)
