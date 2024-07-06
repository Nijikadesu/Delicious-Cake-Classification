import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

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

# dataset = MyDataset('./archive', transform=data_transforms['train'])
# image, label = dataset[1]
# print(image, label)
