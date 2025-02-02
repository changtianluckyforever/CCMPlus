import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms

n_epochs = 3
batch_size = 16
print("torch.cuda.is_available ", torch.cuda.is_available())
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=0)
test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=0)


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(784, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 10)

    def forward(self, din):
        din = din.view(-1, 28 * 28)
        dout = F.relu(self.fc1(din))
        dout = F.relu(self.fc2(dout))
        dout = F.softmax(self.fc3(dout), dim=1)
        return dout


def train():
    model = MLP()
    model.to(device)
    lossfunc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    for epoch in range(n_epochs):
        train_loss = 0.0
        for data, target in train_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = lossfunc(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        train_loss = train_loss / len(train_loader.dataset)
        print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
        # test()
    torch.save(model, "./saved/dqn.th")


def test():
    correct = 0
    total = 0
    num_iter = 0
    start = time.time()
    model = torch.load("./saved/dqn.th")
    model.to(device)
    load_end = time.time()
    print("load model time: ", load_end - start)
    with torch.no_grad():
        for data in test_loader:
            num_iter += 1
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    end = time.time()
    print((end - load_end) / num_iter)
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    return 100.0 * correct / total


if __name__ == '__main__':
    train()
    test()
