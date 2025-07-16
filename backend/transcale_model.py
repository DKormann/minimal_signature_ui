
#%%
from operator import truediv
import PIL.ImageTransform
import torch
from torch import nn


import numpy as np
from pathlib import Path
import PIL
from PIL import Image
from IPython.display import display
# import cv2
import matplotlib.pyplot as plt
from dataclasses import dataclass
# %%

template = Image.open(Path("../template.png"))
[X,Y] = template.size
display(template)




#%%

@dataclass
class transformation:

  # def __init__(self, mat): self.mat = mat

  mat: np.array
  
  @staticmethod
  def new (scale, rotation, translation = (0, 0)):
    sin = np.sin(rotation)
    cos = np.cos(rotation)
    rot_mat = np.array([
      [cos, -sin, 0],
      [sin, cos, 0],
    ])
    mat = rot_mat * scale
    mat[:, 2] = translation
    return transformation(mat)
  

  def inverse(self):
    inv = np.linalg.inv(np.concatenate((self.mat, np.array([[0, 0, 1]])), axis=0))
    return transformation(inv[:2, :])

  def apply_image(self, image):
    return Image.alpha_composite(
      Image.new("RGBA", image.size, "white"),
      image.transform(
        image.size,
        PIL.ImageTransform.AffineTransform(self.mat.flatten()[:6]),
        resample=PIL.Image.Resampling.BICUBIC
      )
    )


tran = transformation.new(1.5, -0.1, (-100, 0))
img = tran.apply_image(template)
display(img)
display(tran.inverse().apply_image(img))





#%%

mat = transformation.new(1, -0.1, (-100,0)).mat
mat = np.concatenate((mat, np.array([[0, 0, 1]])), axis=0)

def create_affine(rotation, scale, translation):
  # if scale.ndim == 1:
  # else:
  #     sx, sy = scale[:, 0], scale[:, 1]

  sx = sy = scale
  cos = torch.cos(rotation)
  sin = torch.sin(rotation)

  a = cos * sx
  b = -sin * sy
  c = translation[:, 0]

  d = sin * sx
  e = cos * sy
  f = translation[:, 1]
  return torch.stack([
      torch.stack([a, b, c], dim=-1),
      torch.stack([d, e, f], dim=-1)
  ], dim=1)  # (B, 2, 3)


def apply_affine(matrix, keypoints):
  N, _ = keypoints.shape
  ones = torch.ones((N, 1), device=keypoints.device, dtype=keypoints.dtype)
  hom_kps = torch.cat([keypoints, ones], dim=-1)  # (N, 3)
  transformed = hom_kps @ matrix.transpose(1, 2)  # -> (N, 2)
  return transformed

def affine_loss(P, Y , width, height, plot=False):
  points = torch.tensor([[0,0],[0,1],[1,0],[1,1],], dtype=torch.float32)*torch.tensor([width, height])

  print(points.shape)
  pred = apply_affine(P, points)
  label = apply_affine(Y, points)
  if plot:
    plt.scatter(label.detach()[0,:,0], label.detach()[0,:,1])
    plt.scatter(pred[0,:,0], pred[0,:,1])


  return (pred-label).square().sum(-1).mean()/torch.tensor([width, height]).square().sum()

def show_points(points):
  points = points.detach().cpu().numpy()
  plt.scatter(points[0,:,0], points[0,:,1])

width, height = template.size

mat = create_affine(torch.tensor([.1]), torch.tensor(1.), torch.tensor([(0,0)]))
mat.requires_grad = True

opt = torch.optim.SGD([mat], lr=0.01)

Y = create_affine(torch.tensor([0.3]), torch.tensor(1.), torch.tensor([(0, 0)]))

#%%

opt.zero_grad()
loss = affine_loss(Y,mat,width,height, 0)
loss.backward()

opt.step()
print(loss.detach().numpy())

#%%


affine_loss(Y, mat, width, height, 1)



#%%

transformed =  template.transform(template.size, PIL.ImageTransform.AffineTransform((
  1.5, 0.5, -100,
  -.5, 1.5, 50
  )) )
transformed = Image.alpha_composite(Image.new("RGBA", transformed.size, "white"), transformed.convert("RGBA"))
display(transformed)


#%%

class BasicBlock(nn.Module):
  def __init__(self, in_c, out_c, stride=1):
    super().__init__()
    self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_c)
    self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_c)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_c != out_c:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_c, out_c, 1, stride, bias=False),
        nn.BatchNorm2d(out_c)
      )

  def forward(self, x):
    out = nn.functional.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)
    return nn.functional.relu(out)


class SmallResNet(nn.Module):
  def __init__(self, num_classes=6):
    super().__init__()
    self.conv = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    self.bn = nn.BatchNorm2d(64)

    specs = [
      (64, 64, 2, 2),
      (64, 100, 2, 4),
      (100, 128, 2, 4),
    ]

    def make_layer(in_c, out_c, num_blocks, stride):
      layers = [BasicBlock(in_c, out_c, stride)]
      for _ in range(1, num_blocks): layers.append(BasicBlock(out_c, out_c))
      return nn.Sequential(*layers)
    
    self.layers = nn.ModuleList(list(map(lambda spec: make_layer(*spec), specs)))
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Linear(specs[-1][1], num_classes)


  def forward(self, x):
    out = nn.functional.relu(self.bn(self.conv(x)))
    for layer in self.layers: out = layer(out)
    out = self.pool(out)
    out = out.view(out.size(0), -1)
    return self.fc(out)

model = SmallResNet()
model = torch.compile(model)

#%%

def step ():

  rot = torch.rand(1).item() - .5
  tran = transformation.new(1., rot, (0, 0))
  img = tran.apply_image(template)
  template_arr = torch.tensor(np.array(img)[:,:,:3])

  print(template_arr.shape)

  template_arr = torch.stack([template_arr, template_arr, template_arr], dim=0)
  print(template_arr.shape)
  
  x = (template_arr).permute(0, 3, 1, 2) / 255.0


  print(x.shape,x.dtype)

  pred=model(x)
  return pred.shape


step()

#%%
