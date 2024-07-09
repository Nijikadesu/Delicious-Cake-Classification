import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),  # 将较小边缘缩放到256像素
        transforms.CenterCrop(224),  # 中心裁剪到224x224
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),  # 将较小边缘缩放到256像素
        transforms.CenterCrop(224),  # 中心裁剪到224x224
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def get_images_and_label(dir_path):
    data = pd.read_csv(os.path.join(dir_path, 'cake_annotated.csv'))
    classes = ['cream', 'fruits', 'sprinkle_toppings']
    images_list = data['file_name']
    labels_list = data[classes]
    return images_list, labels_list


class MyDataset(Dataset):
    def __init__(self, dir_path, transform=None):
        self.dir_path = dir_path
        self.images_list, self.labels_list = get_images_and_label(dir_path)
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dir_path, self.images_list[idx])
        image = Image.open(img_path)
        label = self.labels_list.iloc[idx, :].values
        label = torch.tensor([1 if lbl == 'yes' else 0 for lbl in label], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        else:
            image = np.array(image)
        return image, label

def split_and_load(dir_path, data_transforms=data_transforms):
    dataset = MyDataset(dir_path=dir_path, transform=None)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size - 4
    test_size = 4
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']
    test_dataset.dataset.transform = data_transforms['test']
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_dataset