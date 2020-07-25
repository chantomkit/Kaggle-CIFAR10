from os import listdir
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

l_rate = 0.001
b_size = 4
EPOCHS = 20

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
test = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

trainset = torch.utils.data.DataLoader(train, batch_size=b_size, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=b_size, shuffle=True)

classes = ["PLANE", "CAR", "BIRD", "CAT", "DEER", "DOG", "FROG", "HORSE", "SHIP", "TRUCK"]

def imadd(path):
    img = []
    for i in range(len(path)):
        img.append(transform(cv2.resize(cv2.imread(path[i], 3), (32, 32))).clone().detach().requires_grad_(True).float())
    img = torch.stack([arr for arr in img])
    return img

def imshow(img, label, prediction):

    plt.figure(figsize=(8, 8))
    for i, arr in enumerate(img):
        plt.subplot(2, 2, i + 1)
        image = arr / 2 + 0.5  # unnormalize
        npimg = image.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        plt.title(f"True Class: {classes[int(label[i])]}\nPredicted Class: {classes[int(prediction[i])]}")
    plt.show()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5)            # -> n, 400
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x

net = CNN().to(device)
optimizer = optim.SGD(net.parameters(), lr=l_rate)
loss_function = nn.CrossEntropyLoss()

def train(net):
    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch + 1} / {EPOCHS}")
        for images, labels in tqdm(trainset):
            images = images.to(device)
            labels = labels.to(device)

            print(np.shape(images))
            net.zero_grad()
            outputs = net(images)
            print(outputs, labels)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Loss: {loss}")
    PATH = './CNN.pt'
    torch.save(net.state_dict(), PATH)

def test(net):
    net.load_state_dict(torch.load("CNN.pt"))
    net.eval()
    correct_classes = np.zeros(len(classes), dtype=float)
    total_classes = np.zeros(len(classes), dtype=float)
    with torch.no_grad():
        for images, labels in testset:
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)  # returns a list,
            predicted_class = np.array([])
            for arr in outputs:
                predicted_class = np.append(predicted_class, int(torch.argmax(arr)))

            for i, value in enumerate(predicted_class):
                if value == labels[i]:
                    correct_classes[int(value)] += 1
                total_classes[int(value)] += 1

    correct = np.sum(correct_classes)
    total = np.sum(total_classes)
    acc_classes = np.round(correct_classes / total_classes, 3)
    print(f"Overall accuracy: {round(100*correct/total, 3)}%")
    for i in range(len(classes)):
        print(f"{classes[i]}: {100*acc_classes[i]}%")

def testone(net):
    net.load_state_dict(torch.load("CNN.pt"))
    net.eval()
    with torch.no_grad():
        dataiter = iter(testset)
        images, labels = dataiter.next()
        images = images.to(device)
        labels = labels.to(device)
        # files = listdir("./figused")
        # print(files, type(files))
        # images = imadd([f"./figused/{file}" for file in files])
        # labels = [str(file.split("-")[0]) for file in files]
        outputs = net(images)
        predicted_class = np.array([])
        for i, arr in enumerate(outputs):
            predicted_class = np.append(predicted_class, int(torch.argmax(arr)))
        print(predicted_class, labels)
        imshow(images, labels, predicted_class)

# train(net)
# test(net)
testone(net)