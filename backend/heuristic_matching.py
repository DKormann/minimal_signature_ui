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

img = np.array(scan_preprocessed)
if len(img.shape) == 2 or img.shape[2] == 1: img_with_lines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
else: img_with_lines = img.copy()

def plotline(lines, col):
    for l in lines:  cv2.line(img_with_lines, (l[0], l[1]), (l[2], l[3]), col, 2)

plotline(horlines, (0, 255, 0))
plotline(vertlines, (255, 0, 0))

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
plt.show()

for l in np.concat(template_data, axis=0): plt.plot(*l, c='k')
plt.axis('equal')
plt.show()


#%%


def line_spec(lines):

  ymax = lines[:,:,1].max()
  res = np.zeros((lines[:,1].max() + 1).astype(int))

  lengths = np.abs(lines[:,0, 1] - lines[:,1, 1])

  Ts = lines[:, 1, 0].astype(int)

  for t, l, line in zip(Ts, lengths, lines): res[t] += l
  return res.clip(0, ymax)


vspec = line_spec(rotvlines)
hspec = line_spec(rothlines.swapaxes(1,2))

template_vspec = line_spec(template_data[0])
template_hspec = line_spec(template_data[1].swapaxes(1,2))


plt.plot(template_vspec)

# plt.plot(hspec)

#%%



#%%

X = vspec
T = template_hspec

#%%


def gaussian(size, sigma):
  x = np.arange(size) - size / 2
  g = np.exp(-x**2 / (2 * sigma**2))
  return g / g.sum()

def ckernel (gain, penalty):
  size = penalty*3
  kernel = gaussian(size, gain) - gaussian(size, penalty) / 2
  return kernel / abs(kernel.sum())

def smoothline(s, n=5):
  kernel = np.arange(n)
  kernel = kernel * kernel[::-1]
  kernel = kernel / kernel.sum()
  return np.convolve(s, kernel, mode='same')

def spitz (s):
  res = (s[1:-1] > s[:-2]) * (s[1:-1] > s[2:])
  return np.concatenate([[0], res, [0]])

def clean(s):

  sp = spitz(smoothline(s, 5))
  plt.plot(s)
  plt.show()
  plt.plot(sp)

  return

  n = 100
  ker = gaussian(n, n / 2)
  s = sp / (.0001 + np.convolve(sp, ker, mode='same'))




  plt.plot(s)


clean(hspec)
plt.show()

# clean(template_hspec)

#%%

plt.plot(spitz(template_vspec))

#%%
smooth_n = 20

def tokenize (s):
  s = smoothline(s, smooth_n)
  opt = np.arange(len(s)) [spitz(s) > 0]
  y = s[opt]
  x = np.arange(len(s))[opt]
  return x,y

from dataclasses import dataclass

@dataclass() 
class Samples:
  x: np.ndarray
  y: np.ndarray
  x_size : float
  n : int

  def __init__(self, x, y):
    self.x = x
    self.y = y / (y**2).sum()**0.5
    self.x_size = self.x.max()
    self.n = len(x)
    
  def plot(self, c='b'):
    # plt.plot(self.x, self.y)
    for (x, y, i) in self:
      plt.plot([x, x], [0, y], c=c, lw=1)

  def compare(self, other: 'Samples', dist_penalty = 0.2):
    dist = np.abs(self.x[:, None] - other.x[None, :])
    score = other.y[dist.argmin(1)] * self.y / (dist.min(1) ** 2 * dist_penalty + 1)
    return score.sum()
  
  @staticmethod
  def from_spec(spec: np.ndarray):
    return Samples(*tokenize(spec))

  def mirror(self):
    return Samples(self.x_size - self.x, self.y)
  
  def __iter__(self): return zip(self.x, self.y, range(len(self.x)))


Samples.from_spec(template_vspec).plot()

#%%
@dataclass(frozen=True)
class Transform:
  offset: float
  scale: float
  def apply(self, s:Samples): return Samples ((s.x + self.offset) * self.scale, s.y)

def search_transform(x: Samples, t: Samples):
  best = None
  best_score = -1

  for (tx, _, i) in t:
    for (tx_, _, j) in t:
      if i <= j: continue
      for (ox, _, i) in x:
        for (ox_, _, j) in x:
          if i <= j: continue
          offset = tx - ox
          scale  = (tx_ - tx) / (ox_ - ox)
          if scale <= 0.5 : continue
          if scale >= 2: continue
          p = Transform(offset, scale)
          score = t.compare(p.apply(x), dist_penalty=0.2)
          if score > best_score:
            best_score = score
            best = p
  return best, best_score


x = Samples.from_spec(vspec)
t = Samples.from_spec(template_hspec)

p, score = search_transform(x,t)

t.plot()
p.apply(x).plot('r')

score


#%%


# plt.plot(template_vspec)
Samples.from_spec(template_vspec).plot()

#%%

x = Samples.from_spec(hspec).mirror()
t = Samples.from_spec(template_vspec)


t.plot()
plt.show()
x.plot('r')

#%%

p, score = search_transform(x, t)


#%%

p = Transform(50, 1.25)

t.compare(p.apply(x)), score

#%%

t.plot()
p.apply(x).plot('r')

score