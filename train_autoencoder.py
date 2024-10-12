from model import Autoencoder
from tinygrad import nn, TinyJit
from tinygrad.tensor import Tensor
from dataloader import BoardDataset


x = Tensor.rand(64, 773, requires_grad=False)

@TinyJit
@Tensor.train()
def train(opt, model):
  opt.zero_grad()
  y = model.forward(x)
  loss = x.binary_crossentropy(y).backward()
  opt.step()
  return loss

if __name__ == "__main__":

  model = Autoencoder()
  dataset = BoardDataset("./dataset/dataset_10000.db")
  opt = nn.optim.Adam(nn.state.get_parameters(model))
  for i in range(10): print(train(opt, model).numpy())
