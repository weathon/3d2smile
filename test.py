# In[2]:


import torch, torchvision
import os
import numpy as np
import pylab
import pandas
import sys
from PIL import Image
# In[3]:


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

# In[5]:


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

# from config import get_config
# from eval import Greedy_decode
# from models import build_model
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

# [transformer_.word_map[i] for i in converter.encode("CC(=O)Nc1ccc(O)cc1")]

# In[]
def str_to_vector(s: str)->list:
  return [transformer_.word_map[i] for i in converter.encode(s)]


# In[12]:


import pickle
try:
  with open("Ys.pkl", "rb") as f:
    Ys = pickle.load(f)
except:
  import pandas as pd
  csv = pd.read_csv("/home/wg25r/colab/80k.csv")
  cids = csv["cid"]

  Ys = {}
  invalid_cids = []
  for i in cids.values:
      tmp = str_to_vector(csv[csv["cid"] == i]["canonicalsmiles"].values[0])
      if not tmp == None:
        Ys[i] = tmp
      else:
        invalid_cids.append(i)
  with open("Ys.pkl", "qb") as f:
    pickle.dump(Ys, f)

# In[13]:

print("OOHH")


# In[15]:




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
efficientnetv2 = torchvision.models.efficientnet_v2_l(weights='DEFAULT')
mynet = efficientnetv2.features
#mynet[7] = torch.nn.Identity() this is causing error xkoulatong
class ImageEncoder(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.eff = mynet.to(device)
    self.projection = torch.nn.Linear(1280,256).to(device)
  def forward(self, images): 
    features = self.eff(images)
    features = torch.flatten(features, start_dim=2, end_dim=3)
    features = torch.permute(features, (0, 2, 1))
    return self.projection(features)

# In[27]:


# posem = torch.nn.Embedding(36, 512)
# posem(torch.range(0, 36).unsqueeze(1))

# In[28]:


NUM_HEADS = 4
CHANNELS = [64, 128, 256, 512]
DROPOUT = 0.2
encoder = ImageEncoder() #why this made error on the other side not here becaus ethe paras are in GPU xkou xtiao xkou chouminkunsuankunexk

# In[29]:



# In[30]:




# quit()
# In[31]:


# from torchview import draw_graph

# In[32]:



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
      if next == 77:
        break
      next = torch.sort(out, descending=True)[1][0,0].cpu().detach().numpy()[0] #forgot descending
      conf = conf * softmax(torch.sort(out, descending=True)[0][0].cpu().detach().numpy())[0][0]
      text_in[0] += [next]
    return (text_in[0]), text_in[0], conf

model = Image2SMILES(encoder, transformer)
model.load_state_dict(torch.load("/home/wg25r/main.pb", map_location=device)) #load_state_dict(torch.load("/scratch/st-dushan20-1/eff0_7400.mod")) #kunyunexnaoziyunlewangjileyiweiencoder shiyige nageshilingyigemoxing
gen = SMILESGenerator(encoder, transformer, 128)

model = model.to(device)
gen = gen.to(device)

def softmax(x):
  t = np.exp(x)
  return t/np.sum(t)

from focal_loss.focal_loss import FocalLoss
m = torch.nn.Softmax(dim=-1)
lf = FocalLoss(gamma=2, ignore_index=0)#torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction="none")
def loss_fn(pred, truth):
  # mask = truth != 0
  # pred = pred.permute(0,2,1) #WHY
  # truth = torch.nn.functional.one_hot(truth, num_classes=len(transformer_.word_map.keys())).type(torch.float32).permute(0,2,1)
  pred = m(pred)
  l = lf(pred, truth)
  return l


def mask_acc(pred, truth):
    pred = torch.argmax(pred, -1)
    mask = truth != 0
    match_case = truth == pred
    return torch.sum(mask*match_case)/torch.sum(mask)
# In[57]:


def saveloss(loss_list):
  pylab.clf()
  print(len(loss_list))
  pylab.scatter(np.arange(len(loss_list)), loss_list)
  pylab.plot(np.arange(len(loss_list)), loss_list)
  pylab.savefig("/scratch/st-dushan20-1/cos_loss.png")
  with open("/scratch/st-dushan20-1/cos_loss.txt","w") as f:
    f.write("\n".join([str(i) for i in loss_list]))


_ = list(map(lambda x: np.exp(-0.1*x)+np.random.normal()*0.001*x, list(range(100))))
saveloss(_)

# In[61]:



BATCH_SIZE = 1
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
    # img = np.array(Image.open(f"{HOME_DIR}/rendered/{files[index]}").rotate(np.random.uniform(0,360), expand = 1).resize((400,400)), dtype="float32")
    noise = np.random.uniform(size=img.shape)*20
    img += noise
    return img, [77] + Ys[id], Ys[id] + [78]
    # Xs_img.append(img)
    # Xs_text.append([77] + Ys[id])
    # y.append(Ys[id] + [78])

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
  buffer.put(([Xs_img, pad_pack(Xs_text)[0]], pad_pack(y)[0]))

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

# In[65]:


p = threading.Thread(target=getitem, args=(0,)) #FUCK I KNOW because here is proess touyunchaojieks
# p = multiprocessing.Process(target=getitem, args=(0,)) #FUCK I KNOW because here is proess touyunchaojieks
p.start()
p.join()
buffer.empty()

# In[70]:


# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

# https://stackoverflow.com/questions/51801648/how-to-apply-layer-wise-learning-rate-in-pytorch
optimizer = torch.optim.AdamW(
    #   [
    #    {"params": model.encoder.eff.parameters()},
    #     {"params": model.encoder.mlp.parameters()},
    #     {"params": model.encoder.mha.parameters()},
    #     {"params": model.encoder.norm1.parameters()},
    #     {"params": model.encoder.norm2.parameters()},
    #     {"params": model.encoder.projection.parameters()},
    #     # {"params": model.decoder.parameters(), "lr":5e-8},
    # ]
    model.parameters(),
  #  betas=(0.9999, 0.999),
   lr=0.0003)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, threshold=0.0005, factor=0.4)#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 4000, eta_min=0.0000003, verbose=0)
#.CosineAnnealingLR(optimizer, verbose=True)
#torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.7, verbose=True)


loss_list = []
model.to(device)

#empty queue
while not buffer.empty():
  buffer.get()

p = threading.Thread(target=getitem, args=(0,))
p.start()
p.join()

import wandb








reversed_word_map = {}
for i in transformer_.word_map.keys():
 reversed_word_map[transformer_.word_map[i]] = i

for epoch in range(30):
  np.random.shuffle(files)
  #print('EPOCH {}:'.format(epoch + 1))
  model.train(True)


  start = 0#7400 if epoch == 0 else 0
  for i in range(start, len(files)//BATCH_SIZE-1):
    if i != len(files)//BATCH_SIZE-1:
      p = threading.Thread(target=getitem, args=(i+1,))
      p.start()

    loaded = not buffer.empty()
    if not loaded:
      print("WARNING: reading too slow")
      pass

    (image, text_in), text_out = buffer.get(block=True)
    # if not loaded:
    #   print("Loaded")

    image = image.to(device)
    text_out = text_out.to(device)
    optimizer.zero_grad()
    outputs = model(image, text_in)[0].cpu().detach().numpy()
    print("Teacher forcing Predicted output:\t\t", "".join([reversed_word_map[np.argmax(i)] for i in outputs]))
    print("Token by Token Output: ", "".join([reversed_word_map[np.argmax(i)] for i in gen(image, [[77]])]))
    print("Correct output:\t\t", "".join([reversed_word_map[i] for i in text_out[0].detach().cpu().numpy()]))
    print()

    






