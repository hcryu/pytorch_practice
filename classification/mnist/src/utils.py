import argparse
import wandb

import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torch.nn import functional as F

from src.dataset import get_mnist
from src.model import NeuralNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--devcie", default="cpu", help="device for training")
args = parser.parse_args()

