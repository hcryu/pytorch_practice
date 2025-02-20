import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from train import get_fashion_mnist  # train.py에서 데이터셋 로드 함수 가져오기

def check_data():
    data_dir = './data'
    batch_size = 32

    # FashionMNIST 데이터 로드
    train_data, test_data = get_fashion_mnist(data_dir)

    # 데이터로더 생성
    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=0)

    # 첫 번째 배치 가져오기
    images, labels = next(iter(train_dataloader))

    # 데이터 형태 출력
    print(f"Image batch shape: {images.shape}")  # (batch_size, channels, height, width)
    print(f"Label batch shape: {labels.shape}")  # (batch_size,)

    # 첫 번째 이미지 출력
    plt.figure(figsize=(4, 4))
    plt.imshow(images[0].squeeze(), cmap="gray")
    plt.title(f"Label: {labels[0].item()}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    check_data()
