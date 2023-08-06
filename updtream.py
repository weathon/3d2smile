import torch, torchvision
import os
import numpy as np
import pylab
import pandas
import sys
from PIL import Image

try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

if IN_COLAB:
    HOME_DIR = "/content"
else:
    HOME_DIR = "/arc/project/st-dushan20-1/rendered"


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

import sys
import os
import argparse
from tqdm import tqdm
import deepsmiles
from typing import Any, cast, Callable, List, Tuple, Union
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms


import os
ids = [i.split("_")[0] for i in os.listdir("/arc/project/st-dushan20-1/rendered/rendered")]


import pandas as pd
csv = pd.read_csv("/home/wg25r/colab/80k.csv")
cids = csv["cid"]

Ys = {}
for i in csv:
  Ys[i["cid"]] = i["exactmass"]


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates
    
# https://github.com/Kohulan/DECIMER-Image_Transformer/blob/master/DECIMER/Transformer_decoder.py
import numpy as np
def positional_encoding_1d(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return pos_encoding


def positional_encoding_2d(row, col, d_model):
    assert d_model % 2 == 0
    row_pos = np.repeat(np.arange(row), col)[:, np.newaxis]
    col_pos = np.repeat(np.expand_dims(np.arange(col), 0), row, axis=0).reshape(-1, 1)

    angle_rads_row = get_angles(
        row_pos, np.arange(d_model // 2)[np.newaxis, :], d_model // 2
    )
    angle_rads_col = get_angles(
        col_pos, np.arange(d_model // 2)[np.newaxis, :], d_model // 2
    )

    angle_rads_row[:, 0::2] = np.sin(angle_rads_row[:, 0::2])
    angle_rads_row[:, 1::2] = np.cos(angle_rads_row[:, 1::2])
    angle_rads_col[:, 0::2] = np.sin(angle_rads_col[:, 0::2])
    angle_rads_col[:, 1::2] = np.cos(angle_rads_col[:, 1::2])
    pos_encoding = np.concatenate([angle_rads_row, angle_rads_col], axis=1)[
        np.newaxis, ...
    ]
    return pos_encoding

efficientnetv2 = torchvision.models.efficientnet_v2_s(weights='DEFAULT')
mynet = efficientnetv2.features
mynet[7] = torch.nn.Identity()
class ImageEncoder(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.eff = mynet.to(device)

    self.mlp = torch.nn.Sequential(
        torch.nn.Linear(256,256*2),
        torch.nn.GELU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(256*2, 256),
        torch.nn.GELU(),
    ).to(device)
    self.mha = torch.nn.MultiheadAttention(256, 8, dropout = 0.1).to(device)
    self.norm1 = torch.nn.BatchNorm1d(169).to(device)
    self.norm2 = torch.nn.BatchNorm1d(169).to(device)
    self.projection = torch.nn.Linear(256,256).to(device)
    self.mlp2 = torch.nn.Sequential(
        torch.nn.Linear(43264,2048),
        torch.nn.GELU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(2048, 1024),
        torch.nn.GELU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(1024, 512),
        torch.nn.GELU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(512, 1),
        torch.nn.GELU(),
    ).to(device)
    
  def forward(self, images): 
    features = self.eff(images)
    features = torch.flatten(features, start_dim=2, end_dim=3)
    features = torch.permute(features, (0, 2, 1))
    # print(features.shape)
    pos = positional_encoding_2d(13, 13, 256)
    features = features
    att = self.mha(features, features, features, need_weights=False)[0]
    att = self.norm1(att + features)
    res = self.projection(self.norm2(self.mlp(att) + features))
    f = torch.flatten(res)
    return self.mlp2(f)



NUM_HEADS = 4
CHANNELS = [64, 128, 256, 512]
DROPOUT = 0.2
inp_img = torch.permute(torch.tensor(np.expand_dims(example_in, 0).astype("float32")), (0,3,1,2))
inp_img = inp_img.to(device)
encoder = ImageEncoder()
print(encoder(inp_img).shape)

def softmax(x):
        t = np.exp(x)
        return t/np.sum(t)

from focal_loss.focal_loss import FocalLoss
m = torch.nn.Softmax(dim=-1)
lf = FocalLoss(gamma=0)
def loss_fn(pred, truth):
  return torch.sum((pred-truth)**2)


BATCH_SIZE = 32
files = os.listdir(f"{HOME_DIR}/rendered/")
import multiprocessing, threading
import queue
import time
# data_reader, trainer = multiprocessing.Pipe()
buffer = queue.Queue(maxsize=10) #need maxsize=10, otherwise put will also block
start_index = 0

def process_single(arg):
    _, start_index = arg
    index = start_index + _
    try:
      id = int(files[index].split("_")[0])
    except:
      return
    index = start_index + _
    img = np.array(Image.open(f"{HOME_DIR}/rendered/{files[index]}"), dtype="float32")
    noise = np.random.uniform(size=img.shape)*20
    img += noise
    return img, Ys[id]

def getitem(index):
  start_index = index * BATCH_SIZE
  Xs_img = []
  Xs_text = []
  y = [] #This is slow, rewrite later

  pool = multiprocessing.Pool()
  ans = pool.map(process_single, zip(range(BATCH_SIZE), [start_index]*BATCH_SIZE))
  pool.close()

  Xs_img = [i[0] for i in ans]
  Xs_text = [i[1] for i in ans]
  y = [i[2] for i in ans]
  Xs_img = torch.permute(torch.tensor(np.array(Xs_img)), (0,3,1,2))
  buffer.put([Xs_img, y])

p = threading.Thread(target=getitem, args=(0,)) #It says threading no process, did i used the wrong li
p.start()
print("Started")
# buffer.get()

# In[62]:


t = [[1,'a','b'],
     [1,'a','b'],
     [1,'a','b']]
col_1 = [i[0] for i in t] #gpt
col_1






p = threading.Thread(target=getitem, args=(0,))
# p = multiprocessing.Process(target=getitem, args=(0,))
p.start()
p.join()
buffer.empty()

# In[70]:


# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

# https://stackoverflow.com/questions/51801648/how-to-apply-layer-wise-learning-rate-in-pytorch
optimizer = torch.optim.AdamW(
    model.parameters(),
   lr=0.0007)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1000, verbose=0)
#.CosineAnnealingLR(optimizer, verbose=True)
#torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.7, verbose=True)


loss_list = []
model.to(device)


while not buffer.empty():
  buffer.get()

p = threading.Thread(target=getitem, args=(0,))
p.start()
p.join()

import wandb

wandb.init(
    project="Training 3d to SMILES",
    config={"changes":"Pretrained Model For Images","scheduler": "cus","lr":0.0007,"T0":200,"betas":"0.999,0.9995"}
)


for epoch in range(30):
  np.random.shuffle(files)
  print('EPOCH {}:'.format(epoch + 1))
  model.train(True)
  running_loss = 0.
  last_loss = 0.
  for i in range(len(files)//BATCH_SIZE):
    if i != len(files)//BATCH_SIZE - 1:
      p = threading.Thread(target=getitem, args=(i+1,))
      p.start()

    loaded = not buffer.empty()
    if not loaded:
      print("WARNING: reading too slow")
      pass

    (image, mw) = buffer.get(block=True)

    image = image.to(device)
    mw = mw.to(device)
    optimizer.zero_grad()
    outputs = model(image, text_in)
    loss = loss_fn(outputs, mw)
    loss.backward()

    optimizer.step()

    running_loss += loss.item()
    if i%3==2:
        wandb.log({"loss": running_loss/3, "acc":mask_acc(outputs.detach(), text_out), "lr": optimizer.param_groups[0]['lr']})
        # print(running_loss/10)
        running_loss = 0.
        pass
    if i%20 == 0:
      print(f"Example Output: {gen(inp_img, [[2]])}")
      print(f"Output With Teacher Forcing: {[np.argmax(i) for i in softmax(model(inp_img, [[2]+example_out]).cpu().detach().numpy()[0])]}")
      #iprint(f"Example Output: {(inp_img, [[2]])}")
    if i%10 == 9:
    	scheduler.step()

    if i%100 == 0:
        torch.save(model.state_dict(), f"/scratch/st-dushan20-1/eff_{i}.mod")
