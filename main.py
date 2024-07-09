import torch
import torch.nn as nn
from model import ResNet50FT
from dataset import MyDataset, split_and_load
from visualization import training_curve

device = "cuda" if torch.cuda.is_available() else "cpu"
dir_path = './archive'

train_loader, val_loader, test_loader, test_dataset = split_and_load(dir_path=dir_path)

model = ResNet50FT(classes=3)
model.to(device)

criterion = nn.BCEWithLogitsLoss()

# optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.06)

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

def test(dataloader, model, criterion, class1_acc, class2_acc, class3_acc, avg_ls):
    model.eval()
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    test_loss = 0
    correct = torch.zeros(3).to(device)
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += criterion(pred, y).item()
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()
            correct += (pred == y).float().sum(dim=0)
    test_loss /= num_batches
    correct = correct / size
    accuracy = (100 * correct).cpu().numpy()
    print(f'Test Error: \n    Accuracy: {accuracy}, Avg loss: {test_loss:>8f} \n')
    class1_acc.append(accuracy[0]), class2_acc.append(accuracy[1]), class3_acc.append(accuracy[2])
    avg_ls.append(test_loss)

if __name__ == '__main__':
    epochs = 25
    class1_acc, class2_acc, class3_acc, avg_ls = [], [], [], []
    for t in range(epochs):
        print(f'Epoch {t+1}\n-----------------------------')
        train(train_loader, model, criterion, optimizer)
        test(val_loader, model, criterion, class1_acc, class2_acc, class3_acc, avg_ls)
    print("Done!")
    training_curve(class1_acc, class2_acc, class3_acc, avg_ls)

    pt_path = 'weights_modify.pth'
    torch.save(model.state_dict(), pt_path)
    print(f"model parameters saved to {pt_path}")
