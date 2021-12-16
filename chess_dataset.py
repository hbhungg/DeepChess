import os 
import itertools
import numpy as np

import torch
from torchvision import datasets

class ChessPairDataset(torch.utils.data.Dataset):
  def __init__(self, path, balance=True, train=True, train_split=0.8):
    if train_split < 0 or train_split > 1:
      raise ValueError("Split must be between 0 and 1")
    self.path = path
    self.black_pos = os.listdir(self.path+'black/')
    self.white_pos = os.listdir(self.path+'white/')
    
    # Balance out the 2 datasets
    if balance is True:
      diff = abs(len(self.black_pos) - len(self.white_pos))
      if len(self.black_pos) > len(self.white_pos):
        self.black_pos = self.black_pos[0:len(self.black_pos) - diff]
      else:
        self.white_pos = self.white_pos[0:len(self.white_pos) - diff]
    
    # Get the path from project root
    self.black_pos = ["{}black/{}".format(self.path, i) for i in self.black_pos]
    self.white_pos = ["{}white/{}".format(self.path, i) for i in self.white_pos]

    # Pairwise
    l1 = [(w, b, np.array([1, 0], dtype=float)) for w in self.white_pos for b in self.black_pos][:50000]
    l2 = [(b, w, np.array([0, 1], dtype=float)) for w in self.white_pos for b in self.black_pos][:50000]

    # Split train and test set
    if train is not True:
      l1 = l1[int(len(l1)*train_split):]
      l2 = l2[int(len(l2)*train_split):]
    else:
      l1 = l1[:int(len(l1)*train_split)]
      l2 = l2[:int(len(l2)*train_split)]
    self.all_pair = [*l1, *l2] 
    self.length = len(self.all_pair)

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    pos1, pos2, label = self.all_pair[idx]
    pos1, pos2 = np.load(pos1), np.load(pos2)
    pos1 = torch.Tensor(pos1).type(torch.float)
    pos2 = torch.Tensor(pos2).type(torch.float)
    return pos1, pos2, label



class ChessDataset(torch.utils.data.Dataset):
  def __init__(self, path, train=True, train_split=0.8):
    if train_split < 0 or train_split > 1:
      raise ValueError("Split must be between 0 and 1")
    self.path = path
    self.black_pos = os.listdir("{}black/".format(self.path))
    self.white_pos = os.listdir("{}white/".format(self.path))
    # Get the path from project root
    self.black_pos = ["{}black/{}".format(self.path, i) for i in self.black_pos]
    self.white_pos = ["{}white/{}".format(self.path, i) for i in self.white_pos]

    # Split train and test set
    if train is not True:
      self.black_pos = self.black_pos[int(len(self.black_pos)*train_split):]
      self.white_pos = self.white_pos[int(len(self.white_pos)*train_split):]
    else:
      self.black_pos = self.black_pos[:int(len(self.black_pos)*train_split)]
      self.white_pos = self.white_pos[:int(len(self.white_pos)*train_split)]
    self.dataset = [*self.black_pos, *self.white_pos]
      
    self.length = len(self.dataset)

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    retval = self.dataset[idx]
    retval = np.load(retval)
    retval = torch.Tensor(retval).type(torch.float)
    return retval
