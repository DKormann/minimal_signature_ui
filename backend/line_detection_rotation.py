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



template = np.array(Image.open(Path("./resources/Anwesenheitsliste_lt.png")))[:,:,:3]



# %%

def preprocess(image):
    image = np.array(image.convert("RGB"))
    blurr = cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 5)
    return cv2.Canny(blurr, 50, 150, apertureSize=3)

scan_preprocessed = preprocess(scan)
display(Image.fromarray(scan_preprocessed))


lines = cv2.HoughLinesP(scan_preprocessed, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)


#%%

def guess_rotation(lines):

    if lines is None: return []

    lines = lines[:,0]

    diffs = lines[:,:2] - lines[:,2:]
    length = (diffs**2).sum(axis=1)**0.5

    angles = np.arctan2(diffs[:,1], diffs[:,0]) * 180 / np.pi
    modangles = (angles + 360 + 45) % 9
    
    weighted_mean = np.average(modangles, weights=length) - 45

    corrected_angles = (angles - weighted_mean) % 180
    is_vertical = np.logical_or(corrected_angles < 5, corrected_angles > (180 - 5))
    is_horizontal = np.abs(corrected_angles - 90) < 5


    return lines[is_vertical], lines[is_horizontal], weighted_mean

vertlines, horlines, rotation = guess_rotation(lines)

#%%

img = np.array(scan_preprocessed)
if len(img.shape) == 2 or img.shape[2] == 1: img_with_lines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
else: img_with_lines = img.copy()

def plotline(lines, col):
    for l in lines:  cv2.line(img_with_lines, (l[0], l[1]), (l[2], l[3]), col, 2)

plotline(horlines, (0, 255, 0))
plotline(vertlines, (255, 0, 0))

display(Image.fromarray(img_with_lines))


#%%

def specs(arr):
    print(arr.shape)
    arr = 1-(arr / 255).prod(axis=-1)
    arrv = arr.sum(0)/ arr.shape[0]
    arrh = arr.sum(1) / arr.shape[1]
    return arrv, arrh

rotated = scan.rotate(rotation, fillcolor=(255, 255, 255))
vv, hh = specs(np.array(rotated))


#%%

def rot_lines(lines, angle):
    hlines = lines.reshape(-1, 2, 2)  # [N, 2, 2]    
    angle = - angle / 180 * np.pi
    si = np.sin(angle)
    co = np.cos(angle)
    M = np.array([ [co, -si], [si, co]])
    return (hlines @ M).transpose(0, 2, 1)

rotlines = rot_lines(horlines, -rotation)
for l in rotlines: plt.plot(*l, c='k')

rotlines = rot_lines(vertlines, -rotation)
for l in rotlines: plt.plot(*l, c='k')

plt.axis('equal')

#%%


display(rotated)

#%%

tv, th = specs(template)

plt.plot(hh)
plt.show()
plt.plot(tv)


#%%

plt.plot(vv)
plt.show()
plt.plot(th)




#%%

import torch
hh = hh - hh.mean()

plt.plot(hh)

np.tan(1/15)
# %%
