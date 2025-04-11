# %%
import cv2
import numpy as np
from IPython.display import display
from PIL import Image
import matplotlib.pyplot as plt

from pathlib import Path

scan_fp = "resources/KW_37_bsf-n135_p1.png"
template_fp = "resources/Anwesenheitsliste_lt.png"

sift = cv2.SIFT_create()

def sift_anwesenheitsliste():
    #find matches
    kp1, des1 = sift.detectAndCompute(scan, None)
    kp2, des2 = sift.detectAndCompute(template, None)
    matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=100)).knnMatch(des1, des2, k=2)

    matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    assert len(matches) >= 4, "Not enough matches found"

    # transform matrix
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=10.0)

    projected_image = cv2.warpAffine(src=scan, M=M, dsize=(template.shape[1], template.shape[0]))
    return M, inliers, projected_image

template = np.array(Image.open(template_fp))
scan = np.array(Image.open(scan_fp))

M, inliers, projected_image = sift_anwesenheitsliste()
projected_image = Image.fromarray(projected_image)

projected_image.show()
# projected_image.save("resources/rotated/rotated_signature_scan.png")

