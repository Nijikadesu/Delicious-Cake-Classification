import torch
import torch.nn as nn
from model import ResNet50FT
from torch.utils.data import Dataset, DataLoader, random_split
from dataset import MyDataset, split_and_load

device = "cuda" if torch.cuda.is_available() else "cpu"
dir_path = './archive'

train_loader, val_loader, test_dataset = split_and_load(dir_path=dir_path)

model = ResNet50FT(classes=3)
model.to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9)

def train(dataloader, model, criterion, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 4 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f} [{current:>5d} / {size:>5d}]')

def test(dataloader, model, criterion):
    model.eval()
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    test_loss = 0
    correct = torch.zeros(3).to(device)  # 确保在GPU上计算
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += criterion(pred, y).item()
            pred = torch.sigmoid(pred)  # 将logits转换为概率
            pred = (pred > 0.5).float()  # 将概率转换为二进制标签
            correct += (pred == y).float().sum(dim=0)
    test_loss /= num_batches
    correct = correct / size
    accuracy = (100 * correct).cpu().numpy()
    print(f'Test Error: \n Accuracy: {accuracy}, Avg loss: {test_loss:>8f} \n')

if __name__ == '__main__':
    epochs = 25
    for t in range(epochs):
        print(f'Epoch {t+1}\n-----------------------------')
        train(train_loader, model, criterion, optimizer)
        test(val_loader, model, criterion)
    print("Done!")

    pt_path = 'weights.pth'
    torch.save(model.state_dict(), pt_path)
