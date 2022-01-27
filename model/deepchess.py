import torch
import torch.nn.functional as F
from model.autoencoder import Autoencoder
from model.siamese import Siamese

class DeepChess(torch.nn.Module):
  def __init__(self, ae, si):
    super(DeepChess, self).__init__()
    self.ae = ae
    self.si = si

  def forward(self, x1, x2):
    x1 = self.ae.encode(x1)
    x2 = self.ae.encode(x2)
    x = torch.cat((x1, x2), 1)
    return self.si(x)
