import os
import random

import cv2
import pandas as pd
from tqdm import tqdm

def save_data():
    train_csv_path ='data/fashion-mnist_train.csv'
    test_csv_path = 'data/fashion-mnist_test.csv'

    image_id = 0
    all_labels = []
    for csv_path in [train_csv_path, test_csv_path]:
        root_dir = os.path.dirname(csv_path) # 파일의 경로만 추출 --> data
        os.makedirs(os.path.join(root_dir, 'images'), exist_ok=True)

        data = pd.read_csv(csv_path).to_numpy()
        labels = data[:, 0]
        all_labels.extend(labels)
        images = data[:, 1:]

        for image in tqdm(images):
            cv2.imwrite(os.path.join(root_dir, f'images/{image_id}.jpg'), image.reshape(28, 28, 1))
            image_id += 1

    labels_df = pd.DataFrame(all_labels, columns=['label'])
    labels_df.index.name = 'id'
    labels_df.to_csv(os.path.join(root_dir, 'answers.csv'), index=True)

def split_dataset(csv_path: os.PathLike, split_rate: float = 0.2) -> None:

    root_dir = os.path.dirname(csv_path)

    df = pd.read_csv(csv_path)
    size = len(df)
    indices = list(range(size))

    random.shuffle(indices)

    split_point = int(split_rate * size)

    test_ids = indices[:split_point]
    train_ids = indices[split_point:]

    test_df = df.loc[test_ids]
    test_df.to_csv(os.path.join(root_dir, 'test_answer.csv', index=False))
        # index=False : CSV 파일을 저장할 때 DF의 인덱스를 제외하고 저장
    train_df = df.loc[train_ids]
    train_df.to_csv(os.path.join(root_dir, 'train_answer.csv', index=False))


if __name__=='__main__':
    save_data()
    
