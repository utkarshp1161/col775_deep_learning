# %%
import torch
from resnet import ResNet
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


def resnet_6n_2(n, r):
    """
    This creates a ResNet model with 5 residual blocks 
    in each of the three sets of hidden layers, 
    and a fully connected output layer with 10 units.
    """
    return ResNet(n, r)

resnet = resnet_6n_2(n = 5, r = 10)
#torchvision.datasets.CIFAR10(root='/home/m3rg2000/utkarsh/sym_link/utkarsh_data', download=True)
data_path = '/home/m3rg2000/utkarsh/sym_link/utkarsh_data'


train_dataset = torchvision.datasets.CIFAR10(
    root=data_path,
    train=True,
    download=True,
    transform=ToTensor()
)

test_dataset = torchvision.datasets.CIFAR10(
    root=data_path,
    train=False,
    download=True,
    transform=ToTensor()
)

# %%
# Define data loaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define optimizer and loss function
optimizer = optim.SGD(resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
criterion = nn.CrossEntropyLoss()

num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet.to(device)
for epoch in range(num_epochs):
    resnet.train()
    train_loss = 0
    train_acc = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_acc += torch.sum(preds == labels.data)
    #import pdb as pdb
    #pdb.set_trace()
    train_loss = train_loss / len(train_dataset)
    train_acc = train_acc / len(train_dataset)

    resnet.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = resnet(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            test_acc += torch.sum(preds == labels.data)
    test_loss = test_loss / len(test_dataset)
    test_acc = test_acc / len(test_dataset)

    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'.format(
        epoch+1, num_epochs, train_loss, train_acc, test_loss, test_acc))
    scheduler.step()


# %%



