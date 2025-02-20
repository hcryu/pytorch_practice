from torch import nn, Tensor

class NeuralNetwork(nn.Module):

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),

        )

    def forward(self, x: Tensor) -> Tensor:

        x = self.flatten(x)
        logits = self.linear_relu_stack(x)

        return logits
    
def create_model(num_classes):
    model = NeuralNetwork(num_classes=num_classes)
    return model.cuda()
    # x = torch.randn(4, 1, 28, 28)  # CPU 텐서
    # output = model(x)  # ❌ 오류 발생 (GPU 모델은 GPU 텐서를 기대함)


