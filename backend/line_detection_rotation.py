# %%
import numpy as np
from pathlib import Path
from PIL import Image
from IPython.display import display
import cv2
import matplotlib.pyplot as plt
# %%
scan_fp = Path("resources/KW_37_bsf-n135_p1.png")
scan = Image.open(scan_fp)

display(scan)

# %%

def preprocess(image):
    image = np.array(image.convert("RGB"))
    # image[cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1] > 25] = [255, 255, 255]
    return cv2.Canny(cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 5), 50, 150, apertureSize=3)

scan_preprocessed = preprocess(scan)
display(Image.fromarray(scan_preprocessed))


lines = cv2.HoughLinesP(scan_preprocessed, 1, np.pi / 180, threshold=100, minLineLength=250, maxLineGap=10)


#%%

def guess_rotation(lines):

    if lines is None: return []

    lines = lines[:,0]

    diffs = lines[:,:2] - lines[:,2:]
    length = (diffs**2).sum(axis=1)**0.5

    angles = np.arctan2(diffs[:,1], diffs[:,0]) * 180 / np.pi
    angles = (angles + 360 + 45) % 90

    weighted_mean = np.average(angles, weights=length) - 45

    # plt.hist(angles-45, bins=100)
    # plt.axvline(weighted_mean, color='r', linestyle='dashed', linewidth=1)

    biggest_lines = list(sorted(zip(length, lines.tolist()))) [-40:]
    return [line for _, line in biggest_lines], weighted_mean

scan_lines, rotation = guess_rotation(lines)


#%%

def visualize_lines(lines, img):
    if len(img.shape) == 2 or img.shape[2] == 1: img_with_lines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else: img_with_lines = img.copy()

    for x1, y1, x2, y2 in lines: cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
    display(Image.fromarray(img_with_lines))

visualize_lines(scan_lines, np.array(scan_preprocessed))

print(f"amt of scan lines: {len(scan_lines)}")

#%%


def apply_rotation(img, angle):

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
    return rotated


rotated_image = scan.rotate(rotation)
display(rotated_image)


# # %%


np.tan(1/15)
# %%
