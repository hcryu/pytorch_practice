from torchvision import datasets
from torchvision.transforms import ToTensor
from typing import Tuple

def get_mnist(dir: str = 'data') -> Tuple[datasets.VisionDataset, datasets.VisionDataset]:
    """
    get mnist dataset

    Args:
        dir (str): dataset download direcgtory.

    Returns:
        tuple[datasets.VisionDataset, datasets.VisionDataset]:
            - training_data (dataset.VisionDataset): The MNIST training dataset
            - test_data (datasets.VisionDataset): The MNIST test dataset
    """

    training_data = datasets.MNIST(
        root=dir, # 여기에 지정된 경로로 데이터가 저장됨.
                    # datasets.MNIST 클래스 내부에 dir 폴더를 자동으로 생성하는 기능이 포함되어 있음.
                    # 실행하는 디렉토리 아래 data 폴더가 자동 생성됨.
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root=dir,
        train=False,
        download=True,
        transform=ToTensor()
    )

    return training_data, test_data