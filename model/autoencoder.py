import torch
import torch.nn.functional as F

class Autoencoder(torch.nn.Module):
  def __init__(self):
    super(Autoencoder, self).__init__()
    # Encode
    self.fce1 = torch.nn.Linear(773, 600)
    self.bne1 = torch.nn.BatchNorm1d(600)
    self.fce2 = torch.nn.Linear(600, 400)
    self.bne2 = torch.nn.BatchNorm1d(400)
    self.fce3 = torch.nn.Linear(400, 200)
    self.bne3 = torch.nn.BatchNorm1d(200)
    self.fce4 = torch.nn.Linear(200, 100)
    self.bne4 = torch.nn.BatchNorm1d(100)

    # Decode
    self.fcd1 = torch.nn.Linear(100, 200)
    self.bnd1 = torch.nn.BatchNorm1d(200)
    self.fcd2 = torch.nn.Linear(200, 400)
    self.bnd2 = torch.nn.BatchNorm1d(400)
    self.fcd3 = torch.nn.Linear(400, 600)
    self.bnd3 = torch.nn.BatchNorm1d(600)
    self.fcd4 = torch.nn.Linear(600, 773)
    self.bnd4 = torch.nn.BatchNorm1d(773)

  def encode(self, x):
    x = F.leaky_relu(self.bne1(self.fce1(x)))
    x = F.leaky_relu(self.bne2(self.fce2(x)))
    x = F.leaky_relu(self.bne3(self.fce3(x)))
    x = F.leaky_relu(self.bne4(self.fce4(x)))
    return x

  def decode(self, x):
    x = F.leaky_relu(self.bnd1(self.fcd1(x)))
    x = F.leaky_relu(self.bnd2(self.fcd2(x)))
    x = F.leaky_relu(self.bnd3(self.fcd3(x)))
    x = torch.sigmoid(self.bnd4(self.fcd4(x)))
    return x

  def forward(self, x):
    x = self.encode(x)
    return self.decode(x), x

