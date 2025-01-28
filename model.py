import math
from tinygrad import nn, Tensor

class Linear:
  def __init__(self, in_features:int, out_features:int, bias=True):
    self.weight = Tensor.glorot_uniform(out_features, in_features)
    self.bias = Tensor.glorot_uniform(out_features) if bias else None
  def __call__(self, x:Tensor) -> Tensor:
    return x.linear(self.weight.transpose(), self.bias)

class Autoencoder:
  def __init__(self):
    self.encode_layers = [
      Linear(773, 600), Tensor.leakyrelu,
      Linear(600, 400), Tensor.leakyrelu,
      Linear(400, 200), Tensor.leakyrelu,
      Linear(200, 100), Tensor.leakyrelu
    ]
    self.decode_layers = [
      Linear(100, 200), Tensor.leakyrelu,
      Linear(200, 400), Tensor.leakyrelu,
      Linear(400, 600), Tensor.leakyrelu,
      Linear(600, 773)
    ]

  def encode(self, x: Tensor):
    return x.sequential(self.encode_layers)
  def decode(self, x: Tensor):
    return x.sequential(self.decode_layers)
  def forward(self, x: Tensor):
    return self.decode(self.encode(x))

class DeepChess:
  def __init__(self, ae:Autoencoder):
    # self.ae = ae
    # Siamese
    self.layers = [
      *ae.encode_layers, lambda x: x.reshape(-1, 100*2),
      Linear(200, 400), nn.BatchNorm(400), Tensor.leakyrelu,
      Linear(400, 200), nn.BatchNorm(200), Tensor.leakyrelu,
      Linear(200, 100), nn.BatchNorm(100), Tensor.leakyrelu,
      Linear(100, 2), lambda x: Tensor.softmax(x, axis=1)
    ]
  def forward(self, x:Tensor):
    return x.sequential(self.layers)

