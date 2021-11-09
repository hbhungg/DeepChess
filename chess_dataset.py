import os 
import itertools
import torch
import numpy as np
from torchvision import datasets

class ChessDataset(torch.utils.data.Dataset):
  def __init__(self, path, balance=True):
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
    l1 = [(w, b, np.array([1, 0])) for w in self.white_pos for b in self.black_pos]
    l2 = [(b, w, np.array([0, 1])) for w in self.white_pos for b in self.black_pos]
    self.all_pair = [*l1, *l2] 

  def __len__(self):
    return len(self.all_pair)

  def __getitem__(self, idx):
    pos1, pos2, label = self.all_pair[idx]
    return np.load(pos1), np.load(pos2), label

if __name__ == "__main__":
  cc = ChessDataset('dataset/npy/')
  print(len(cc))
  print(len(cc.black_pos))
  print(len(cc.white_pos))
