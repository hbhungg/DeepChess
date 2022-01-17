import torch
import torch.nn.functional as F


class Siamese(torch.nn.Module):
  def __init__(self):
    super(Siamese, self).__init__()
    self.fc1 = torch.nn.Linear(200, 400)
    self.bn1 = torch.nn.BatchNorm1d(400)
    self.fc2 = torch.nn.Linear(400, 200)
    self.bn2 = torch.nn.BatchNorm1d(200)
    self.fc3 = torch.nn.Linear(200, 100)
    self.bn3 = torch.nn.BatchNorm1d(100)
    self.fc4 = torch.nn.Linear(100, 2)
    self.bn4 = torch.nn.BatchNorm1d(2)

  def forward(self, x):
    x = F.leaky_relu(self.bn1(self.fc1(x)))
    x = F.leaky_relu(self.bn2(self.fc2(x)))
    x = F.leaky_relu(self.bn3(self.fc3(x)))
    x = self.bn4(self.fc4(x))
    #return x
    return F.softmax(x, dim=1)

