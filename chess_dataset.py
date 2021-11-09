import os 
import itertools
import torch
import numpy as np
from torchvision import datasets

class ChessDataset(torch.utils.data.Dataset):
  def __init__(self, chess_dir, balance=True):
    self.black_pos = os.listdir(chess_dir+'black/')
    self.white_pos = os.listdir(chess_dir+'white/')
    
    # Balance out the 2 datasets
    if balance is True:
      diff = abs(len(self.black_pos) - len(self.white_pos))
      if len(self.black_pos) > len(self.white_pos):
        self.black_pos = self.black_pos[0:len(self.black_pos) - diff]
      else:
        self.white_pos = self.white_pos[0:len(self.white_pos) - diff]

    self.all_pair = list(itertools.product(self.black_pos, self.white_pos)) 
  
  def __len__(self):
    return len(self.all_pair)

  def __getitem__(self, idx):
    pair = self.all_pair[idx]

if __name__ == "__main__":
  cc = ChessDataset('dataset/npy/game_states/')
  print(len(cc))
  print(len(cc.black_pos))
  print(len(cc.white_pos))
