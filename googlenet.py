#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
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
transform = transforms.Compose([transforms.Resize((80)), transforms.ToTensor(), transforms.Normalize((0.5),(0.5))])

#--- Dividing the datasets in training and testing sets --------------------------------------
train = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
trainLoad = torch.utils.data.DataLoader(dataset=train, batch_size=32, shuffle=True)
testLoad = torch.utils.data.DataLoader(dataset=test, batch_size=32, shuffle=True)

#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

class Inception(nn.Module):
    def __init__(self, in_channel, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.norm1_1 = nn.BatchNorm2d(in_channel, eps=1e-3)
        self.p1_1 = nn.Conv2d(in_channels=in_channel, out_channels=c1, kernel_size=(1,1))
        self.norm2_1 = nn.BatchNorm2d(in_channel, eps=1e-3)
        self.p2_1 = nn.Conv2d(in_channels=in_channel, out_channels=c2[0], kernel_size=(1,1))
        self.norm2_2 = nn.BatchNorm2d(c2[0], eps=1e-3)
        self.p2_2 = nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=(3,3), padding=(1,1))
        self.norm3_1 = nn.BatchNorm2d(in_channel, eps=1e-3)
        self.p3_1 = nn.Conv2d(in_channels=in_channel, out_channels=c3[0], kernel_size=(1,1))
        self.norm3_2 = nn.BatchNorm2d(c3[0], eps=1e-3)
        self.p3_2 = nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=(5,5), padding=(2,2))
        self.p4_1 = nn.MaxPool2d(kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.norm4_2 = nn.BatchNorm2d(in_channel, eps=1e-3)
        self.p4_2 = nn.Conv2d(in_channels=in_channel, out_channels=c4, kernel_size=(1,1))

    def forward(self, x):
        p1 = self.p1_1(F.relu(self.norm1_1(x)))
        p2 = self.p2_2(F.relu(self.norm2_2(self.p2_1(F.relu(self.norm2_1(x))))))
        p3 = self.p3_2(F.relu(self.norm3_2(self.p3_1(F.relu(self.norm3_1(x))))))
        p4 = self.p4_2(F.relu(self.norm4_2(self.p4_1(x))))
        return torch.cat((p1, p2, p3, p4), dim=1)

#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

class GoogleNet(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(GoogleNet, self).__init__()
        layers = []
        layers += [nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3)),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))]
        layers += [nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1)),
                   nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3,3), padding=(1,1)),
                   nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))]
        layers += [Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))]
        layers += [Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))]
        layers += [Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AvgPool2d(kernel_size=2)]
        self.net = nn.Sequential(*layers)
        self.dense = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 1024 * 1 * 1)
        x = self.dense(x)
        return x

#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

model = GoogleNet(1, 10).to(device)
loss = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
cost = 0
epochs = 25
iterations = []
trainLoss = []
trainAcc = []
testLoss = []
testAcc = []
totalTime = time.time()

for i in range(epochs):
    epochTime = time.time()
    if i==0:
        print("------------------------------------------------")
        print("------------------------------------------------")
        print("Starting...")
        print("Learning Rate: {}".format(learning_rate))
        print("Starting with: {}".format(epochs))
        print("------------------------------------------------")
        print("------------------------------------------------")
    print("At epoch: " + str(i+1) + "...")

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
plt.title('GoogleNet - Accuracies')
plt.plot(trainAcc, color='red', label='Train Accuracy')
plt.plot(testAcc, color='blue', label='Validation Accuracy')
plt.legend()
plt.savefig('/Users/8fan/DLCV_CW2/mini/Graphs/GoogleNet/Accuracy_{}_{}.png'.format(str(learning_rate).replace('.',''), epochs))

#--- Plotting the losses ---------------------------------------------------------------------
plt.figure(figsize=(10, 7))
plt.title('GoogleNet - Losses')
plt.plot(trainLoss, color='red', label='Train Loss')
plt.plot(testLoss, color='blue', label='Validation Loss')
plt.legend()
plt.savefig('/Users/8fan/DLCV_CW2/mini/Graphs/GoogleNet/Loss_{}_{}.png'.format(str(learning_rate).replace('.',''), epochs))

#--- Plotting the heatmap --------------------------------------------------------------------
axis = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
heatmap = pd.DataFrame(data=0, index=axis, columns=axis)
with torch.no_grad():
    for images, labels in testLoad:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(16):
            true_label = labels[i].item()
            predicted_label = predicted[i].item()
            heatmap.iloc[true_label, predicted_label] += 1
_, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(heatmap, annot=True, fmt='d', cmap='YlGn')
figure = ax.get_figure()
figure.savefig('/Users/8fan/DLCV_CW2/mini/Graphs/GoogleNet/Heatmap_{}_{}.png'.format(str(learning_rate).replace('.',''), epochs), dpi=400)

#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------