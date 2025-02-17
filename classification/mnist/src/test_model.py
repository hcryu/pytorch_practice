import torch
from model import NeuralNetwork

num_classes = 10
model = NeuralNetwork(num_classes=num_classes)

print(model)

# 임의의 입력값 생성
dummy_input = torch.randn(1, 1, 28, 28) # (batch_size, channels, height, width)

output = model(dummy_input)

print(f"Output shape: {output.shape}")
