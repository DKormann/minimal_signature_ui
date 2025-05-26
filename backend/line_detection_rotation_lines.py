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

template = Image.open(Path("./resources/Anwesenheitsliste_lt.png"))


display (template)
# %%

def preprocess(image):
  image = np.array(image.convert("RGB"))
  blurr = cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 5)
  return cv2.Canny(blurr, 50, 150, apertureSize=3)

scan_preprocessed = preprocess(scan)
temp_preprocessed = preprocess(template)


display(Image.fromarray(scan_preprocessed))
display(Image.fromarray(temp_preprocessed))

lines = cv2.HoughLinesP(scan_preprocessed, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
template_lines = cv2.HoughLinesP(temp_preprocessed, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

#%%

def guess_rotation(lines):

  if lines is None: return []

  lines = lines[:,0]

  diffs = lines[:,:2] - lines[:,2:]
  length = (diffs**2).sum(axis=1)**0.5

  angles = np.arctan2(diffs[:,1], diffs[:,0]) * 180 / np.pi
  modangles = (angles + 360 + 45) % 90
  
  weighted_mean = np.average(modangles, weights=length) - 45

  corrected_angles = (angles - weighted_mean) % 180
  is_vertical = np.logical_or(corrected_angles < 5, corrected_angles > (180 - 5))
  is_horizontal = np.abs(corrected_angles - 90) < 5

  return lines[is_vertical], lines[is_horizontal], weighted_mean

vertlines, horlines, rotation = guess_rotation(lines)

template_data = guess_rotation(template_lines)


print(vertlines.shape)
#%%

img = np.array(scan_preprocessed)
if len(img.shape) == 2 or img.shape[2] == 1: img_with_lines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
else: img_with_lines = img.copy()

def plotline(lines, col):
    for l in lines:  cv2.line(img_with_lines, (l[0], l[1]), (l[2], l[3]), col, 2)

plotline(horlines, (0, 255, 0))
plotline(vertlines, (255, 0, 0))

#%%


def rot_lines(lines, angle):
  hlines = lines.reshape(-1, 2, 2)  # [N, 2, 2]    
  angle = - angle / 180 * np.pi
  si = np.sin(angle)
  co = np.cos(angle)
  M = np.array([ [co, -si], [si, co]])
  return (hlines @ M).transpose(0, 2, 1)

rothlines = rot_lines(horlines, -rotation)
rotvlines = rot_lines(vertlines, -rotation)

template_data = (rot_lines(template_data[0],0), rot_lines(template_data[1],0))

for l in np.concat([rotvlines, rothlines], axis=0): plt.plot(*l, c='k')


plt.axis('equal')


#%%



for l in template_data[0]: plt.plot(*l, c='k', )
for l in template_data[1]: plt.plot(*l, c='k', )



#%%


def line_spec(lines):

  res = np.zeros((lines[:,1].max() + 1).astype(int))

  lengths = np.abs(lines[:,0, 1] - lines[:,1, 1])

  Ts = lines[:, 1, 0].astype(int)
  print(Ts.shape)
  for t, l, line in zip(Ts, lengths, lines): res[t] += l
  return res


vspec = line_spec(rotvlines)
hspec = line_spec(rothlines.swapaxes(1,2))

template_vspec = line_spec(template_data[0])
template_hspec = line_spec(template_data[1].swapaxes(1,2))


plt.plot(vspec)
plt.show()
plt.plot(hspec)
plt.show()
plt.plot(template_hspec)
plt.show()
plt.plot(template_vspec)

#%%


X = vspec
T = template_hspec

#%%

plt.plot(X)
plt.plot(T)

#%%

def smoothline(s, n=5):
  kernel = np.arange(n)
  kernel = kernel * kernel[::-1]
  kernel = kernel / kernel.sum()
  return np.convolve(s, kernel, mode='same')


SX = smoothline(X, 10)
ST = smoothline(T, 10)



def rescale_signal(signal, new_length):
    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, new_length)
    return np.interp(x_new, x_old, signal)

#%%

plt.plot(rescale_signal(SX[55:], 2850))
plt.plot(ST)

#%%

plt.plot(rescale_signal(SX[::-1][100:], 2800))
plt.plot(ST)
