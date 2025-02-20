import argparse

import lightning as L
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.dataset import get_fashion_mnist
from src.model import create_model

from pytorch_lightning.loggers import WandbLogger
import wandb



L.seed_everything(36, workers=True)

class DatasetModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.trainset, self.validset = get_fashion_mnist(dir="./data")

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, 
                          drop_last=True, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False, 
                          drop_last=True, pin_memory=True)
    

class LitTrainingModule(L.LightningModule):
    def __init__(self, model, lr) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs):
        return self.model(inputs)
    
    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        train_loss = self.loss_fn(outputs, targets)

    # 로그 추가 (학습 손실)
        self.log("train_loss", train_loss, prog_bar=True, on_step=True, on_epoch=True)

        return train_loss
    
    def on_training_epoch_end():
        pass

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        valid_loss = self.loss_fn(outputs, targets)

    # 로그 추가 (검증 손실)
        self.log("val_loss", valid_loss, prog_bar=True, on_step=True, on_epoch=True)

        # 터미널에 직접 출력 (추가)
        # 터미널 출력은 100 step마다만
        if batch_idx % 100 == 0:
            print(f"\n[Validation] Step: {batch_idx}, val_loss: {valid_loss.item()}", end="")
    
        return valid_loss
    
    def on_validation_epoch_end(self):
        #print("on_validation_epoch_end() has been called!!")
        avg_val_loss = self.trainer.callback_metrics.get("val_loss", torch.tensor(0.0))
        
        if not self.trainer.callback_metrics:
            print("callback_metrics is empty. Validation might not be running.")
            return
        
        print()
        # 확인용 로그 출력
        print(f"callback_metrics: {self.trainer.callback_metrics}")  
    
    # 터미널 출력 추가
        print(f"Epoch {self.current_epoch} - avg_val_loss: {avg_val_loss.item()}")

    # 로그 추가 (평균 검증 손실)
        self.log("avg_val_loss", avg_val_loss, prog_bar=True, on_epoch=True)
        


def run(args):
    num_classes = 10
    devices = args.devices
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    num_workers = args.num_workers

    #if len(args.devices) == 1:
    #    devices = [int(args.devices)]
    #else:
    #    devices = list(map(int, args.devices.split(',')))

    if isinstance(args.devices, list):  # 리스트 형태인 경우 (nargs='+' 사용 시)
        devices = list(map(int, args.devices))
    elif isinstance(args.devices, str):  # 문자열 형태인 경우 ("0,1" 입력 처리)
        devices = list(map(int, filter(None, args.devices.replace(" ", "").split(','))))  
    else:  # 단일 값 처리
        devices = [int(args.devices)]


    dataset = DatasetModule(batch_size=batch_size, num_workers=num_workers)
    model = LitTrainingModule(create_model(num_classes), lr=lr)

    #wandb.init(project="fashion-mnist", name="lightning_run")  # ✅ 명시적 초기화

    # 🟢 WandbLogger 추가
    wandb_logger = WandbLogger(project="fashion-mnist", name="lightning_run")

    trainer = L.Trainer(
            check_val_every_n_epoch=1,
            log_every_n_steps=10,
            max_epochs=epochs,
            accelerator='gpu',
            devices=devices,
            precision='16-mixed',
            logger=wandb_logger,  # ✅ wandb_logger를 Trainer에 추가!
    )

    trainer.fit(model=model, train_dataloaders=dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', type=str, nargs='+', default="0")
    parser.add_argument("--batch_size", type=int, default=32, help="학습 및 검증에 사용할 배치 크기")
    parser.add_argument("--epochs", type=int, default=10, help="학습 epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="학습 및 검증에 사용할 worker 수")
    args = parser.parse_args()
    run(args)
