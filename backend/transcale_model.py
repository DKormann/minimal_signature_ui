
#%%
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
# %%

template = Image.open(Path("../template.png"))
[X,Y] = template.size
display(template)


#%%

np.concat

#%%

class transformation:

  def __init__(self, mat): self.mat = mat
  
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

#%%


transformed =  template.transform(template.size, PIL.ImageTransform.AffineTransform((
  1.5, 0.5, -100,
  -.5, 1.5, 50
  
  )) )
transformed = Image.alpha_composite(Image.new("RGBA", transformed.size, "white"), transformed.convert("RGBA"))
display(transformed)


#%%


def create_XY(k = 100):
  """k is the size of the training snippets"""

  t_k = min(template_arr.size)
  
  max_scale = max(0.6, t_k / k)
  scale = 
  
  resized = template.resize(())

create_XY()

#%%

template_arr.shape # (848, 1361, 3)

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
  def __init__(self, num_classes=3):
    super().__init__()
    self.conv = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    self.bn = nn.BatchNorm2d(64)
    
    self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=2)
    self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=4)
    # self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)

    self.pool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Linear(128, num_classes)

  def _make_layer(self, in_c, out_c, num_blocks, stride):
    layers = [BasicBlock(in_c, out_c, stride)]
    for _ in range(1, num_blocks):
        layers.append(BasicBlock(out_c, out_c))
    return nn.Sequential(*layers)

  def forward(self, x):
    out = nn.functional.relu(self.bn(self.conv(x)))
    out = self.layer1(out)
    out = self.layer2(out)
    # out = self.layer3(out)
    out = self.pool(out)
    out = out.view(out.size(0), -1)
    return self.fc(out)


model = SmallResNet()


print(sum(p.numel() for p in model.parameters()))

x = torch.tensor(template_arr, dtype=torch.float32).permute( 2, 0, 1).unsqueeze(0) / 255.0
print(x.shape)

#%%

plt.imshow(x[0].permute(1,2,0)[200:400,200:400].detach().numpy())

#%%


pred = model(x[:,:,:1000,:1000])
pred.shape

#%%
