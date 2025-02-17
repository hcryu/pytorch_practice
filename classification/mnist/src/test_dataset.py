from dataset import get_mnist

# MNIST 데이터셋 로드 테스트
train_data, test_data = get_mnist()

# 데이터셋 크기 확인
print(f"Training set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")


# 첫 번째 이미지와 라벨 확인
image, label = train_data[0]
print(f"First image shpae: {image.shape}, Label: {label}")