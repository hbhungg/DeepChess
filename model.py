from tinygrad import nn, Tensor

class Autoencoder():
  def __init__(self):
    self.encode_layers = [
      nn.Linear(773, 600), nn.BatchNorm(600), Tensor.leakyrelu,
      nn.Linear(600, 400), nn.BatchNorm(400), Tensor.leakyrelu,
      nn.Linear(400, 200), nn.BatchNorm(200), Tensor.leakyrelu,
      nn.Linear(200, 100), nn.BatchNorm(100),
    ]
    self.decode_layers = [
      nn.Linear(100, 200), nn.BatchNorm(200), Tensor.leakyrelu,
      nn.Linear(200, 400), nn.BatchNorm(400), Tensor.leakyrelu,
      nn.Linear(400, 600), nn.BatchNorm(600), Tensor.leakyrelu,
      nn.Linear(600, 773), nn.BatchNorm(773),
    ]
  def encode(self, x: Tensor):
    return x.sequential(self.encode_layers)
  def decode(self, x: Tensor):
    return x.sequential(self.decode_layers)
  def forward(self, x: Tensor):
    return self.decode(self.encode(x))

  # def forward(self, x):
  #   x = self.encode(x)
  #   return self.decode(x), x


class Siamese():
  def __init__(self):
    self.layers = [
      nn.Linear(200, 400), nn.BatchNorm(400), Tensor.leakyrelu,
      nn.Linear(400, 200), nn.BatchNorm(200), Tensor.leakyrelu,
      nn.Linear(200, 100), nn.BatchNorm(100), Tensor.leakyrelu,
      nn.Linear(100, 2), nn.BatchNorm(2),
      lambda x: Tensor.softmax(x, axis=1)
    ]

  def forward(self, x):
    return x.sequential(self.layers)


class DeepChess():
  def __init__(self, ae, si):
    self.ae = ae
    self.si = si

  def forward(self, x1, x2):
    x1 = self.ae.encode(x1)
    x2 = self.ae.encode(x2)
    x = Tensor.cat(x1, x2, dim=1)
    return self.si(x)

