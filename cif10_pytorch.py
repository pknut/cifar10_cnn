"""
accuracy training   = 78.614%
accuracy validation = 63.51%
"""
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from deepsense import neptune

ctx = neptune.Context()
ctx.tags.append('cifar10')
ctx.tags.append('fine')
ctx.tags.append('pytorch')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

epochs = 20


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.fc1 = nn.Linear(in_features=32 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # batch_size x 3 x 32 x 32
        x = self.pool(F.relu(self.conv1(x)))
        # batch_size x 16 x 14 x 14
        x = self.pool(F.relu(self.conv2(x)))
        # batch_size x 32 x 5 x 5
        x = x.view(-1, 32 * 5 * 5)
        # batch_size x 32*5*5
        x = F.relu(self.fc1(x))
        # batch_size x 120
        x = F.relu(self.fc2(x))
        # batch_size x 84
        x = self.fc3(x)
        #batch_size x 10
        return x


net = Net()
net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print('Start')

for epoch in range(epochs):

    correct = 0
    total = 0

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        inputs, labels = data

        inputs, labels = Variable(inputs), Variable(labels)
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()

    # total=5000, i+1=12500

    print('Epoch: ', epoch+1, '/', epochs)
    ctx.channel_send('Log-loss training', epoch + 1, running_loss/(i+1))
    ctx.channel_send('Accuracy training', epoch + 1, correct / total)

    if (epoch+1) % 3 == 0:
        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = net(Variable(images))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        ctx.channel_send('Accuracy validation', epoch + 1, correct / total)


print('Finished Training')

correct = 0
total = 0
for data in testloader:
    images, labels = data
    images, labels = images.cuda(), labels.cuda()
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (correct / total))
ctx.channel_send('Accuracy validation', epochs, correct / total)

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    images, labels = images.cuda(), labels.cuda()
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], class_correct[i] / class_total[i]))

