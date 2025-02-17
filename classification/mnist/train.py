import argparse
import wandb

import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torch.nn import functional as F

from src.dataset import get_mnist
from src.model import NeuralNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", help="device for training")
args = parser.parse_args()

def log_test_predictions(images, labels, outputs, predicted, test_table, log_counter) -> None:
    
    scores = F.softmax(outputs.detach(), dim=1)
    log_scores = scores.cpu().numpy()
    log_images = images.cpu().numpy()
    log_labels = labels.cpu().numpy()
    log_preds = predicted.cpu().numpy()

    _id = 0

    for i, l, p, s in zip(log_images, log_labels, log_preds, log_scores):
        img_id = str(_id) + "_" +str(log_counter)
        test_table.add_data(img_id, wandb.Image(i), p, l, *s)

def train_one_epoch(dataloader: DataLoader, device: str, model: nn.Module, loss_fn: nn.Module, optimizer: optim.Optimizer, epoch: int) -> None:
    
    size = len(dataloader.dataset)
    model.train()

    for batch, (images, targets) in enumerate(dataloader):

        images = images.to(device)
        targets = targets.to(device)
        targets = torch.flatten(targets)

        preds = model(images)
        loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            wandb.log({"train_loss": loss, "epoch": epoch})
            loss = loss.item()
            current = batch * len(images)
            print(f'loss: {loss:>7f} [{current:5d}/{size:>5d}]')

def valid_one_epoch(dataloader, device, model, loss_fn, epoch, test_table) -> None:

    size = len(dataloader.dataset)
    num_batches = len(dataloader) # 실제 train() 함수에서 배치크기를 지정함. 이 크기를 바탕으로 배치 개수를 결정함
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch, (images, targets) in enumerate(dataloader):

            images = images.to(device)
            targets = targets.to(device)
            targets = torch.flatten(targets)

            preds = model(images)

            test_loss += loss_fn(preds, targets).item()
            correct += (preds.argmax(1) == targets).float().sum().item() # float() : boolean을 실수값으로 변환

            if batch == 0:
                log_test_predictions(images, targets, preds, preds.argmax(1), test_table, epoch)

    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')
    wandb.log({"test_loss": test_loss, "test_accuracy": correct, "epoch": epoch})


def train(device) -> None:

    num_classes = 10
    batch_size = 32
    epochs = 10
    lr = 1e-3

    data_dir = 'data'
    train_data, test_data = get_mnist(data_dir)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    model = NeuralNetwork(num_classes=num_classes).to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    test_table = wandb.Table(columns = ["id", "image", "predicted", "true", *[f"class_{i}_score" for i in range(10)]])

    for t in range(epochs):
        print(f'Epoch {t+1}\n------------------------------------')
        train_one_epoch(train_loader, device, model, loss_fn, optimizer, t+1)
        valid_one_epoch(test_loader, device, model, loss_fn, t+1, test_table)
    print("Done!")


    wandb.log({"predictions": test_table})
    torch.save(model.state_dict(), 'mnist-net.pth')
    print('Saved Pytorch Model State to mnist-net.pth')

if __name__ == "__main__":
    wandb.init(
        project="mnist_with_wndb",
        config={
            "learning_rate": 1e-3,
            "architecture": "NeuralNetwork",
            "dataset": "MNIST",
            "epochs": 10,
        }
    )
    train(device=args.device)
    wandb.finish()






    
