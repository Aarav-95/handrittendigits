# Make sure to set the intepreter path to Anaconda or MiniConda before running the program to ensure no errors
# Import necessary modules
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
import helper
from torchvision import datasets, transforms
from six.moves import urllib

# Allows for data to be downloaded 
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# Transforms the data into tensors and normalizes to reduce redundant data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])

trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Neural Network Model Structure with different nodes in each layer
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

# Negative Log Likelihood Loss is defined to calculate loss
criterion = nn.NLLLoss()

# Optimizer to train neural network with a defined learning rate
optimizer = optim.SGD(model.parameters(), lr=0.1)

# number of times network is trained
epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # TODO: Training pass
        
        # Clear the gradients, do this because gradients are accumulated
        optimizer.zero_grad()

        # Forward pass, then backward pass, then update weights
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        # Take an update step and view the new weights
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")

dataiter = iter(trainloader)
images, labels = next(dataiter)

images = images.view(images.shape[0], -1)
logps = model(images)

# Output of the network are log-probabilities, need to take exponential for probabilities
prob = torch.exp(logps)

x = True 
label_iterations = 0
prob_iterations = 0
labels_new = labels
prob_new = prob

torch.set_printoptions(sci_mode=False)

# While loop to display results
while x == True:
    for labelsnum in labels_new:
        print(labelsnum)
        break   
    for probnum in prob_new:
        print(probnum)
        break
    label_iterations += 1 
    labels_new = labels[label_iterations:]
    prob_iterations += 1
    prob_new = prob[prob_iterations:]

    if label_iterations == 64 and prob_iterations == 64:
        x == False  
