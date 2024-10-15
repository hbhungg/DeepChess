import math
from model import Autoencoder
from tinygrad import nn, TinyJit
from tinygrad.tensor import Tensor
from dataloader import BoardDataset
import mlflow
from mlflow.data.dataset import Dataset
from tinygrad.nn.state import safe_save, get_state_dict
from tinygrad.nn.optim import Optimizer
from tqdm import tqdm

@TinyJit
@Tensor.train()
def train(opt:Optimizer, model:Autoencoder, x:Tensor):
  opt.zero_grad()
  y = model.forward(x)
  loss = (y.binary_crossentropy(x, reduction="mean")).backward()
  opt.step()
  return loss