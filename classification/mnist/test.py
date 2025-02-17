import argparse

import torch
from torch import nn
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from src.dataset import get_mnist
from src.model import NeuralNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", help="device for test")
args = parser.parse_args()

def predict(test_data, model, device) -> None:
    
    model.eval()

    image = test_data[0][0]
    plt.imshow(image.permute(1,2,0))
    plt.savefig("output_image.png")
    image = image.to(device)
    image = image.unsqueeze(0) # (C, H, W) → (1, C, H, W) 형태로 변환, 배치를 위한 차원 추가.
                                # 모델에 입력되는 매트릭스 형태를 동일하게 하기 위하여 필요함.
    target = test_data[0][1]
    target = torch.tensor(target)

    with torch.no_grad():
        pred = model(image)
        predicted = pred[0].argmax(0) # image의 크기: (1, C, H, W) → 모델 입력
                                        # pred = model(image)의 출력(pred)은 일반적으로 (1, num_classes) 형태임.
                                        # pred[0]을 하면 (10,) 차원의 텐서가 나옴.
                                        # argmax(0)을 하면 가장 큰 값을 가지는 인덱스(클래스 번호)를 반환.
        actual = target
        print(f'Predicted: "{predicted}", Actual: "{actual}"')

def test(device) -> None:
    
    num_classes = 10

    data_dir = 'data'
    _, test_data = get_mnist(data_dir)

    model = NeuralNetwork(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('mnist-net.pth'))

    predict(test_data, model, device)


if __name__ == "__main__":
    test(device=args.device)