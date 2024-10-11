from model import Autoencoder
from tinygrad.tensor import Tensor


@Tensor.train()
def train():
    model = Autoencoder()
    # opt = nn.optim.Adam(nn.state.get_parameters(model))
    x = Tensor.rand(2, 773, requires_grad=False)
    y = model.forward(x)
    print(x.numpy())
    print(y.numpy())
    print(x.binary_crossentropy(y).backward().numpy())

if __name__ == "__main__":
  train()
