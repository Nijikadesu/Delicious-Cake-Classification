import torch
from model import ResNet50FT
import matplotlib.pyplot as plt
from main import test_loader, test_dataset
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pt_path = 'weights.pth'
dir_path = './archive'

def model_infer(pt_path):
    model = ResNet50FT(classes=3)
    model.load_state_dict(torch.load(pt_path))
    model.to(device)
    model.eval()
    return model

def result_infer(model, dataloader):
    model = model.to(device)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    idx = 0
    for (X, y) in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        pred = (pred > 0.5).float()
        image_dir = test_dataset.dataset.get_image_path(test_dataset.indices[idx])
        image = Image.open(image_dir)
        axes[idx].imshow(image)
        axes[idx].set_title(f"Inference result: {pred.cpu().numpy()} / Ground truth: {y.cpu().numpy()}")
        axes[idx].axis('off')
        idx += 1

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    model  = model_infer(pt_path=pt_path)
    result_infer(model, test_loader)
