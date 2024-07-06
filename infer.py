import torch
import torch.nn as nn
from model import ResNet50FT

def model_infer(pt_path):
    model = ResNet50FT(classes=3)
    model.load_state_dict(torch.load(pt_path))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
