#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
cuda = torch.device('cuda')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

#--- Resize the images to 80x80 pixels -------------------------------------------------------
transform_train = transforms.Compose([transforms.Resize((80)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5),(0.5))])
transform_test = transforms.Compose([transforms.Resize((80)), transforms.ToTensor(), transforms.Normalize((0.5),(0.5))])

#--- Dividing the datasets in training and testing sets --------------------------------------
train = torchvision.datasets.MNIST(root='./data', train=True, transform=transform_train, download=True)
test = torchvision.datasets.MNIST(root='./data', train=False, transform=transform_test, download=True)
trainLoad = torch.utils.data.DataLoader(dataset=train, batch_size=32, shuffle=True)
testLoad = torch.utils.data.DataLoader(dataset=test, batch_size=32, shuffle=False)

#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1,1), downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(3,3), stride=stride, padding=(1,1), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

class ResNet34(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=(2,2))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=(2,2))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=(2,2))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=(1,1)):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                                       nn.BatchNorm2d(planes))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # 224x224
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 112x112

        x = self.layer1(x)  # 56x56
        x = self.layer2(x)  # 28x28
        x = self.layer3(x)  # 14x14
        x = self.layer4(x)  # 7x7

        x = self.avgpool(x)  # 1x1
        x = torch.flatten(x, 1)  # remove 1 X 1 grid and make vector of tensor shape
        x = self.fc(x)

        return x


def resnet34():
    layers = [3, 4, 6, 3]
    model = ResNet34(Bottleneck, layers)
    return model

#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

model = resnet34()
model = model.cuda()
loss = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
cost = 0
epochs = 10
iterations = []
trainLoss = []
trainAcc = []
testLoss = []
testAcc = []
totalTime = time.time()

for i in range(epochs):
    epochTime = time.time()
    if i == 0:
        print("------------------------------------------------")
        print("------------------------------------------------")
        print("Starting...")
        print("Learning Rate: {}".format(learning_rate))
        print("Starting with: {}".format(epochs))
        print("------------------------------------------------")
        print("------------------------------------------------")
    print("At epoch: " + str(i + 1) + "...")

    model.train()
    correct = 0
    for X, Y in trainLoad:
        X = X.to(cuda)
        Y = Y.to(cuda)
        optimizer.zero_grad()
        hypo = model(X)
        cost = loss(hypo, Y)
        cost.backward()
        optimizer.step()
        prediction = hypo.data.max(1)[1]
        correct += prediction.eq(Y.data).sum()

    model.eval()
    correct2 = 0
    for data, target in testLoad:
        data = data.to(cuda)
        target = target.to(cuda)
        output = model(data)
        cost2 = loss(output, target)
        prediction = output.data.max(1)[1]
        correct2 += prediction.eq(target.data).sum()

    iterations.append(i)
    trainLoss.append(cost.tolist())
    testLoss.append(cost2.tolist())
    trainAcc.append((100*correct/len(trainLoad.dataset)).tolist())
    testAcc.append((100*correct2/len(testLoad.dataset)).tolist())
    epochTime = time.time() - epochTime

    print("Train Accuracy: " + str(trainAcc[i]))
    print("Test Accuracy: " + str(testAcc[i]))
    print("Epoch time elapsed: ", epochTime)
    print("------------------------------------------------")
    print("------------------------------------------------")

totalTime = time.time() - totalTime
print("---")
print("Total time elapsed: ", totalTime)
print("---")
print("------------------------------------------------")
print("------------------------------------------------")

#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

#--- Plotting the accuracies -----------------------------------------------------------------
plt.figure(figsize=(10, 7))
plt.title('ResNet - Accuracies')
plt.plot(trainAcc, color='red', label='Train Accuracy')
plt.plot(testAcc, color='blue', label='Validation Accuracy')
plt.legend()
plt.savefig('/Users/8fan/DLCV_CW2/mini/Graphs/ResNet/Accuracy_{}_{}.png'.format(str(learning_rate).replace('.',''), epochs))

#--- Plotting the losses ---------------------------------------------------------------------
plt.figure(figsize=(10, 7))
plt.title('ResNet - Losses')
plt.plot(trainLoss, color='red', label='Train Loss')
plt.plot(testLoss, color='blue', label='Validation Loss')
plt.legend()
plt.savefig('/Users/8fan/DLCV_CW2/mini/Graphs/ResNet/Loss_{}_{}.png'.format(str(learning_rate).replace('.',''), epochs))

#--- Plotting the heatmap --------------------------------------------------------------------
axis = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
heatmap = pd.DataFrame(data=0,index=axis,columns=axis)
with torch.no_grad():
    for images, labels in testLoad:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(16):
            true_label = labels[i].item()
            predicted_label = predicted[i].item()
            heatmap.iloc[true_label,predicted_label] += 1
_, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(heatmap, annot=True, fmt='d',cmap='YlGn')
figure = ax.get_figure()
figure.savefig('/Users/8fan/DLCV_CW2/mini/Graphs/ResNet/Heatmap_{}_{}.png'.format(str(learning_rate).replace('.',''), epochs), dpi=400)

#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------