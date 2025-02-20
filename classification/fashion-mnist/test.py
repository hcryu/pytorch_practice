import argparse

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

#from src.dataset import FashionMnistDataset
from src.dataset import get_fashion_mnist

from src.model import NeuralNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", help='Device for Learning')
args = parser.parse_args()

def predict(test_data, model, device):

    classes = [
        'T-shirt/top',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle boot',
    ]

    model.eval()
    data_num = 1 # data selection
    image = test_data[data_num][0].to(device)
        # test_data[data_num][0] 는 image 텐서 -> .to(device) 가능
    image = image.unsqueeze(0)
    #target = test_data[0][1].to(device)
            # test_data[data_num][1] 은 label 정수형값 -> .to(device) 불가능
            # to(device)는 텐서(tensor)에 대해서만 동작

    target = test_data[data_num][1]
    target = torch.tensor(target).to(device)

    with torch.no_grad():
        pred = model(image)
        predicted = classes[pred[0].argmax(0)]
        actual = classes[target]
        print(f'Predicted "{predicted}", Actual: "{actual}"')

def test(device):
    image_dir = 'data/fashion-mnist/images'
    test_csv_path = 'data/fashion-mnit/test_answer.csv'
    data_dir ='data'

    num_classes = 10

    #test_data = FashionMnistDataset(
    #    image_dir,
    #    test_csv_path
    #)
    _, test_data = get_fashion_mnist(data_dir)

    model = NeuralNetwork(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('fashion-mnist-net.pth', weights_only=True))

    predict(test_data, model, device)

if __name__=='__main__':
    test(args.device)