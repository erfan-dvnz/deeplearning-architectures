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

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.layer_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))

        self.layer_2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))

        self.layer_3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))

        self.layer_4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))

        self.layer_5 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))
        height, width = 7, 7
        self.layer_6 = nn.Sequential(nn.Linear(512 * width * height, 4096),
                                     nn.ReLU(True),
                                     nn.Dropout(p=0.5),
                                     nn.Linear(4096, 4096),
                                     nn.ReLU(True),
                                     nn.Dropout(p=0.5),
                                     nn.Linear(4096, 10))
        for m in self.modules():
            if isinstance(m, torch.torch.nn.Conv2d) or isinstance(m, torch.torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.detach().zero_()

        self.avgpool = torch.nn.AdaptiveAvgPool2d((height, width))

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.layer_6(x)
        return x

#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

model = VGG16()
model = model.cuda()
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
    if i == 0:
        print("------------------------------------------------")
        print("------------------------------------------------")
        print("Starting with:")
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
plt.title('VGG16 - Accuracies')
plt.plot(trainAcc, color='red', label='Train Accuracy')
plt.plot(testAcc, color='blue', label='Validation Accuracy')
plt.legend()
plt.savefig('/Users/8fan/DLCV_CW2/mini/Graphs/VGG16/Accuracy_{}_{}.png'.format(str(learning_rate).replace('.',''), epochs))

#--- Plotting the losses ---------------------------------------------------------------------
plt.figure(figsize=(10, 7))
plt.title('VGG16 - Losses')
plt.plot(trainLoss, color='red', label='Train Loss')
plt.plot(testLoss, color='blue', label='Validation Loss')
plt.legend()
plt.savefig('/Users/8fan/DLCV_CW2/mini/Graphs/VGG16/Loss_{}_{}.png'.format(str(learning_rate).replace('.',''), epochs))

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
figure.savefig('/Users/8fan/DLCV_CW2/mini/Graphs/VGG16/Heatmap_{}_{}.png'.format(str(learning_rate).replace('.',''), epochs), dpi=400)

#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------