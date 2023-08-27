#!/usr/bin/env python
# coding: utf-8

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

# In[4]:


sys.path.append("/home/wg25r/with_pretrain/IUPAC2Struct")

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# From the SwinOCSR
import sys
sys.path.append("/home/wg25r/with_pretrain/SwinOCSR/model/Swin-transformer-focalloss")
sys.path.append("/home/wg25r/with_pretrain/SwinOCSR/model/")

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

from pre_transformer import Transformer


class FocalLossModelInference:
    """
    Inference Class
    """
    def __init__(self):
        # Load dictionary that maps tokens to integers
        word_map_path = '/home/wg25r/with_pretrain/SwinOCSR/Data/500wan/500wan_shuffle_DeepSMILES_word_map'
        self.word_map = torch.load(word_map_path)
        self.inv_word_map = {v: k for k, v in self.word_map.items()}

        # Define device, load models and weights
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        # self.args, config = self.get_inference_config()
        # self.encoder = build_model(config, tag=False)
        self.decoder = self.build_decoder()
        self.load_checkpoint("/home/wg25r/with_pretrain/swin_transform_focalloss.pth")
        self.decoder = self.decoder.to(self.dev).eval()
        # self.encoder = self.encoder.to(self.dev).eval()

    def load_checkpoint(self, checkpoint_path):
        """
        Load checkpoint and update encoder and decoder accordingly

        Args:
            checkpoint_path (str): path of checkpoint file
        """
        print(f"=====> Resuming from {checkpoint_path} <=====")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # encoder_msg = self.encoder.load_state_dict(checkpoint['encoder'],
        #                                            strict=False)
        decoder_msg = self.decoder.load_state_dict(checkpoint['decoder'],
                                                   strict=False)
        # print(f"Encoder: {encoder_msg}")
        print(f"Decoder: {decoder_msg}")
        del checkpoint
        torch.cuda.empty_cache()

    def build_decoder(self):
        """
        This method builds the Transformer decoder and returns it
        """
        self.decoder_dim = 256  # dimension of decoder RNN
        self.ff_dim = 2048
        self.num_head = 8
        self.dropout = 0.1
        self.encoder_num_layer = 6
        self.decoder_num_layer = 6
        self.max_len = 277
        self.decoder_lr = 5e-4
        self.best_acc = 0.
        return Transformer(dim=self.decoder_dim,
                           ff_dim=self.ff_dim,
                           num_head=self.num_head,
                           encoder_num_layer=self.encoder_num_layer,
                           decoder_num_layer=self.decoder_num_layer,
                           vocab_size=len(self.word_map),
                           max_len=self.max_len,
                           drop_rate=self.dropout,
                           tag=False)
transformer_ = FocalLossModelInference()
transformer = transformer_.build_decoder().decoder

# In[ ]:
print(transformer_.word_map)


print(device)

# In[6]:

converter = deepsmiles.Converter(rings=True, branches=True)


# In[]
def str_to_vector(s: str)->list:
  return [transformer_.word_map[i] for i in converter.encode(s)]

# In[]:


import os
ids = [i.split("_")[0] for i in os.listdir("/arc/project/st-dushan20-1/rendered/rendered")]


# In[10]:


import pandas as pd
csv = pd.read_csv("/home/wg25r/colab/80k.csv")
cids = csv["cid"]
csv.columns

# In[11]:


len(cids)

# In[12]:



Ys = {}
invalid_cids = []
for i in cids.values:
    tmp = str_to_vector(csv[csv["cid"] == i]["canonicalsmiles"].values[0])
    if not tmp == None:
      Ys[i] = tmp
    else:
      invalid_cids.append(i)

# In[13]:

if len(invalid_cids) == 0:
  print("OOHH")

# In[14]:


example_in = Image.open(f"{HOME_DIR}/rendered/6912034_0.jpg")
example_out = csv[csv["cid"]==6912034]["canonicalsmiles"].values[0]

# In[15]:


example_out = str_to_vector(example_out)


torch.permute(torch.tensor(np.expand_dims(example_in, 0).astype("float32")), (0,3,1,2)).size()

# In[23]:


t = torch.randn(1,64,400,400)
torch.flatten(t)
torch.flatten(t, start_dim=2,end_dim=3).shape

# In[24]:

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
# In[]
efficientnetv2 = torchvision.models.efficientnet_v2_m(weights='DEFAULT')
mynet = efficientnetv2.features
#mynet[7] = torch.nn.Identity() this is causing error xkoulatong
class ImageEncoder(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.eff = mynet.to(device)

    self.mlp = torch.nn.Sequential(
        torch.nn.Linear(1280,2048),
        torch.nn.GELU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(2048, 1280),
        torch.nn.GELU(),
    ).to(device)
    self.mha = torch.nn.MultiheadAttention(1280, 8, dropout = 0.1).to(device)
    self.norm1 = torch.nn.BatchNorm1d(169).to(device)
    self.norm2 = torch.nn.BatchNorm1d(169).to(device)
    self.projection = torch.nn.Linear(1280,256).to(device)
  def forward(self, images): 
    features = self.eff(images)
    pos = positional_encoding_2d(13, 13, 1280)
    # features = pos + features #FUCK I FORGOT ADD POS yesterday too tired
    features = torch.flatten(features, start_dim=2, end_dim=3)
    features = torch.permute(features, (0, 2, 1))
    # print(features.shape)
    features = features + torch.tensor(pos, dtype=torch.float32).to(device)
    att = self.mha(features, features, features, need_weights=False)[0]
    att = self.norm1(att + features)
    return self.projection(self.norm2(self.mlp(att) + features))

# In[28]:


NUM_HEADS = 8
CHANNELS = [64, 128, 256, 512]
DROPOUT = 0.2
inp_img = torch.permute(torch.tensor(np.expand_dims(example_in, 0).astype("float32")), (0,3,1,2))
inp_img = inp_img.to(device)
encoder = ImageEncoder() #why this made error on the other side not here becaus ethe paras are in GPU xkou xtiao xkou chouminkunsuankunexk
#encoder = torch.load("/scratch/st-dushan20-1/eff_9900.mod")
#encoder.load_state_dict(torch.load("/scratch/st-dushan20-1/effnet7_5000.mod"))
# print(encoder(inp_img).shape)

# In[29]:


sum(p.numel() for p in encoder.parameters())

# In[30]:


from torchinfo import summary
print(summary(encoder, (1,3,400,400), depth=10))


inp_img = inp_img.to(device)

for p in transformer.parameters():
  p.requires_grad = False

sum(p.numel() for p in transformer.parameters() if p.requires_grad)

# In[45]:
# from another paper
def pad_pack(sequences):
    maxlen = max(map(len, sequences))
    batch = torch.LongTensor(len(sequences),maxlen).fill_(0)
    for i,x in enumerate(sequences):
        batch[i,:len(x)] = torch.LongTensor(x)
    return batch, maxlen

# https://github.com/suanfaxiaohuo/SwinOCSR/blob/main/model/Swin-transformer-focalloss/pre_transformer.py#L95
def triangle_mask(size):
    mask = 1- np.triu(np.ones((1, size, size)),k=1).astype('uint8')
    mask = torch.autograd.Variable(torch.from_numpy(mask))
    return mask


class Image2SMILES(torch.nn.Module):
  def __init__(self, encoder, decoder):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, image, text_in):
    padded_text, maxlen = pad_pack(text_in)
    padded_text = padded_text.to(device)
    image_feature = self.encoder(image)
    out = self.decoder(padded_text, image_feature, x_mask=triangle_mask(maxlen).to(device))
    return out

# In[46]:

def softmax(x):
        t = np.exp(x)
        return t/np.sum(t)
class SMILESGenerator(torch.nn.Module):
  def __init__(self, encoder, decoder, max_len):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.max_len = max_len

  def forward (self, image, text_in):
    image_feature = self.encoder(image)
    conf = 1
    for i in range(self.max_len):
      padded_text = pad_pack(text_in)[0]
      padded_text = padded_text.to(device)
      out = self.decoder(padded_text, image_feature, x_mask=triangle_mask(len(text_in)).to(device))
      # out = self.generator(out)
      next = torch.sort(out, descending=True)[1][0,0].cpu().detach().numpy()[0] #forgot descending
      conf = conf * softmax(torch.sort(out, descending=True)[0][0].cpu().detach().numpy())[0][0]
      #if next == 3:
      #    text_in.append(3)
      #    break
      text_in[0] += [next]
      # print(text_in)
    return (text_in[0]), text_in[0], conf

model = Image2SMILES(encoder, transformer)
model.load_state_dict(torch.load("/scratch/st-dushan20-1/effnet_medium4_3000.mod"))
gen = SMILESGenerator(encoder, transformer, 128)

model = model.to(device)
gen = gen.to(device)

print(torch.argmax(model(inp_img, [[2, 0]])[0,0]))
print("output shape", (model(inp_img, [[2, 0]])).shape)

# In[50]:


gen(inp_img, [[77]])
# In[51]:


def softmax(x):
  t = np.exp(x)
  return t/np.sum(t)

# In[52]:


pylab.plot(softmax(model(inp_img, [[2, 0]])[0][0].cpu().detach().numpy()))
# pylab.savefig("test.png")


sum(p.numel() for p in model. parameters() if not p.requires_grad)


import torch

from focal_loss.focal_loss import FocalLoss
m = torch.nn.Softmax(dim=-1)
lf = FocalLoss(gamma=2, ignore_index=0)#torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction="none")
def loss_fn(pred, truth):
  pred = m(pred)
  l = lf(pred, truth)
  return l


def mask_acc(pred, truth):
    pred = torch.argmax(pred, -1)
    mask = truth != 0
    match_case = truth == pred
    return torch.sum(mask*match_case)/torch.sum(mask)



import pickle

_ = list(map(lambda x: np.exp(-0.1*x)+np.random.normal()*0.001*x, list(range(100))))
saveloss(_)


BATCH_SIZE = 16
files = os.listdir(f"{HOME_DIR}/rendered/")
files = [i for i in files if len(i)<=40]
import multiprocessing, threading
import queue
import time
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
    img = np.array(Image.open(f"{HOME_DIR}/rendered/{files[index]}").rotate(np.random.uniform(0,360), expand = 1).resize((400,400)), dtype="float32")
    noise = np.random.uniform(size=img.shape)*20
    img += noise
    return img, [77] + Ys[id], Ys[id] + [78]

def getitems(s, e):
 for index in range(s, e):
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
  buffer.put(([Xs_img.to(device), pad_pack(Xs_text)[0].to(device)], pad_pack(y)[0].to(device)))


print("Started")
# buffer.get()

# In[62]:


t = [[1,'a','b'],
     [1,'a','b'],
     [1,'a','b']]
col_1 = [i[0] for i in t] #gpt
col_1

# In[63]:


next(zip([1,2,3],[0,0,0]))

# In[64]:


# https://superfastpython.com/multiprocessing-pool-for-loop/
# naoziyunle def douwnagjiel
def task(x):
  return x+1 if x%3!=0 else None

pool = multiprocessing.Pool()
ans = pool.map(task, range(10))
pool.close()
list(filter(lambda x: not x is None, ans))

buffer.empty()

# In[70]:


# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

# https://stackoverflow.com/questions/51801648/how-to-apply-layer-wise-learning-rate-in-pytorch
optimizer = torch.optim.AdamW(
    model.parameters(),
   lr=0.00025)


scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99986)



loss_list = []
model.to(device)

#empty queue
while not buffer.empty():
  buffer.get()



import wandb

wandb.init(
    project="Training 3d to SMILES",
    config={"changes":"eff with att Pretrained Model For Images","scheduler": "cus","lr":0.0007,"T0":200,"betas":"0.999,0.9995"}
)


for epoch in range(30):
  np.random.shuffle(files)
  print('EPOCH {}:'.format(epoch + 1))
  model.train(True)
  running_loss = 0
  last_loss = 0
  start = 3001 if epoch == 0 else 0
  p = threading.Thread(target=getitems, args=(start, len(files)//BATCH_SIZE-1))
  p.start() 
  for i in range(start, len(files)//BATCH_SIZE-1):
    loaded = not buffer.empty()
    if not loaded:
      print("WARNING: reading too slow")


    (image, text_in), text_out = buffer.get(block=True)

    image = image
    text_out = text_out
    optimizer.zero_grad()
    outputs = model(image, text_in)
    loss = loss_fn(outputs, text_out)
    loss.backward()

    optimizer.step()

    running_loss += loss.item()

    if i%20==19:
        wandb.log({"loss": running_loss/20, "acc":mask_acc(outputs.detach(), text_out), "lr": optimizer.param_groups[0]['lr']})
        scheduler.step()
        running_loss = 0.
    if i%200 == 0:
      print(f"Output With Teacher Forcing: {[np.argmax(i) for i in softmax(model(inp_img, [[77]+example_out]).cpu().detach().numpy()[0])]}")
    

    if i%500 == 0:
        torch.save(model.state_dict(), f"/scratch/st-dushan20-1/effnet_medium{epoch}_{i}.mod")
