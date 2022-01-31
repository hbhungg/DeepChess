import numpy as np

import torch
from torchvision import datasets


class ChessPairDataset(torch.utils.data.Dataset):
  def __init__(self, white_wins, black_wins, perspective=True, train=True, train_split=0.8, length=1000000) :
    """
    Pytorch dataset handler

    Parameters:
      white_wins: 
    """
    if train_split < 0 or train_split > 1:
      raise ValueError("Split must be between 0 and 1")
    self.black_wins = black_wins
    self.white_wins = white_wins

    # Split train and test set
    if train is not True:
      self.white_wins = self.white_wins[int(len(self.white_wins) * train_split):]
      self.black_wins = self.black_wins[int(len(self.black_wins) * train_split):]
    else:
      self.white_wins = self.white_wins[:int(len(self.white_wins) * train_split)]
      self.black_wins = self.black_wins[:int(len(self.black_wins) * train_split)]
    
    self.length = length
    self.perspective = perspective

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    if self.perspective is True:
      rand_win = self.white_wins[np.random.randint(0, len(self.white_wins))]
      rand_loss = self.black_wins[np.random.randint(0, len(self.black_wins))]
    else:
      rand_win = self.black_wins[np.random.randint(0, len(self.black_wins))]
      rand_loss = self.white_wins[np.random.randint(0, len(self.white_wins))]
      

    order = np.random.randint(0, 2)
    if order == 0:
      rand_win = torch.from_numpy(rand_win).type(torch.FloatTensor)
      rand_loss = torch.from_numpy(rand_loss).type(torch.FloatTensor)
      label = torch.from_numpy(np.array([1, 0])).type(torch.FloatTensor)
      return rand_win, rand_loss, label
    else:
      rand_win = torch.from_numpy(rand_win).type(torch.FloatTensor)
      rand_loss = torch.from_numpy(rand_loss).type(torch.FloatTensor)
      label = torch.from_numpy(np.array([0, 1])).type(torch.FloatTensor)
      return rand_loss, rand_win, label
      

class ChessDataset(torch.utils.data.Dataset):
  def __init__(self, white_wins, black_wins, train=True, train_split=0.8):
    if train_split < 0 or train_split > 1:
      raise ValueError("Split must be between 0 and 1")
    
    self.white_wins = white_wins
    self.black_wins = black_wins

    # Split train and test set
    if train is not True:
      self.white_wins = self.white_wins[int(len(self.white_wins) * train_split):]
      self.black_wins = self.black_wins[int(len(self.black_wins) * train_split):]
      self.p = list(range(0, len(self.white_wins) + len(self.black_wins)))
    else:
      self.white_wins = self.white_wins[:int(len(self.white_wins) * train_split)]
      self.black_wins = self.black_wins[:int(len(self.black_wins) * train_split)]
      self.p = list(range(0, len(self.white_wins) + len(self.black_wins)))

    # An array of length black_wins + white_wins
    # Then shuffle it, since shuffle the orignal array is very costly in term of memory and time
    np.random.shuffle(self.p)
    self.length = len(self.p)

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    # Real index
    idx = self.p[idx]
    if idx >= len(self.white_wins):
      idx = idx - len(self.white_wins)
      return torch.from_numpy(self.black_wins[idx]).type(torch.FloatTensor)
    else:
      return torch.from_numpy(self.white_wins[idx]).type(torch.FloatTensor)
