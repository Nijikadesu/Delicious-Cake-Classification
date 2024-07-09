import torch
import os
from model import ResNet50FT
import matplotlib.pyplot as plt
from dataset import MyDataset, data_transforms, get_images_and_label
from main import test_dataset
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pt_path = 'model_weights.pth'
dir_path = './archive'

def model_infer(pt_path):
    model = ResNet50FT(classes=3)
    model.load_state_dict(torch.load(pt_path))
    model.to(device)
    model.eval()
    return model

def result_infer(model, transform=None):
    model = model.to(device)
    dataset = MyDataset(dir_path=dir_path, transform=transform)
    images_list, _ = get_images_and_label(dir_path=dir_path)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    for i in range (0, 4):
        image, label = dataset.__getitem__(0)
        image = image.unsqueeze(0).to(device)
        pred = model(image)
        pred = (pred > 0.5).float()
        image_dir = images_list[i]
        image_dir = os.path.join('./archive/', image_dir)
        image = Image.open(image_dir)
        axes[i].imshow(image)
        axes[i].set_title(f"Inference result: {pred}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    model  = model_infer(pt_path=pt_path)
    result_infer(model, transform=data_transforms['val'])
