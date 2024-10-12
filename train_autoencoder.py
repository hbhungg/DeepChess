from model import Autoencoder
from tinygrad import nn, TinyJit
from tinygrad.tensor import Tensor
from dataloader import BoardDataset


x = Tensor.rand(64, 773, requires_grad=False)

@TinyJit
@Tensor.train()
def train(opt, model, x:Tensor):
  opt.zero_grad()
  y = model.forward(x)
  loss = (y.binary_crossentropy(x, reduction="mean")).backward()
  opt.step()
  return loss

if __name__ == "__main__":

  model = Autoencoder()
  dataset = BoardDataset("./dataset/dataset_10000.db")
  opt = nn.optim.Adam(nn.state.get_parameters(model), lr=1e-4)

  EPOCH = 1
  BATCH_SIZE = 30
  ls = []

  for i in range(EPOCH): 
    for x in range(0, len(dataset), BATCH_SIZE):
      X = dataset[x:x+BATCH_SIZE-1][0]
      if X.shape[0] != BATCH_SIZE: continue # STOOPID
      loss = train(opt, model, X).item()
      ls.append(loss)
      print(loss)

  import matplotlib.pyplot as plt

  plt.plot(ls)
  plt.show()
