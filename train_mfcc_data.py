import random
import numpy as np
import torch
import torchvision
from torch import nn
from torch.optim import SGD
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import one_hot

import csv

from NN import Network


def make_data_set(file_path):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    labels = []

    with open('./files/data.txt', 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            data = [float(l) for l in line[:-1]]
            label = line[-1]

            if label not in labels:
                labels.append(label)

            if random.randint(0, 9) > 6:
                train_data.append(data)
                train_labels.append(labels.index(label))
            else:
                test_data.append(data)
                test_labels.append(labels.index(label))
    
    return train_data, train_labels, test_data, test_labels, labels




    # max_label = 0
    # X, labels = [], []
    # for line in lines:
    #     if not line:
    #         continue
    #     line = line.split(' ')
    #     datas = line[:-1]
    #     x = []
    #     for data in datas:
    #         x.append(float(data))
    #     X.append(x)

    #     label = int(line[-1])
    #     labels.append(label)
    #     if max_label < label:
    #         max_label = label

    # data_set = SupervisedDataSet(13, max_label + 1)
    # for data, label in zip(X, labels):
    #     label_data = np.zeros(max_label + 1).astype(np.int32)
    #     label_data[int(label)] = 1
    #     data_set.addSample(data, label_data)

    # return data_set


# print 'read train dataset'
# train_data_set = make_data_set('doc/train.txt')

# print 'read test dataset'
# test_data_set = make_data_set('doc/test.txt')
print('load dataset')
train_x, train_y, test_x, test_y, classes = make_data_set('./files/data.txt')

print('convert data to tensor')
tensor_train_x = torch.Tensor(train_x)
# tensor_train_y = torch.Tensor(train_y)
tensor_train_y = one_hot(torch.tensor(train_y), num_classes=len(classes))
train_set = TensorDataset(tensor_train_x, tensor_train_y)
train_loader = DataLoader(train_set, batch_size=10, shuffle=True, num_workers=0)

tensor_test_x = torch.Tensor(test_x)
# tensor_test_y = torch.Tensor(test_y)
tensor_test_y = one_hot(torch.tensor(test_y), num_classes=len(classes))
test_set = TensorDataset(tensor_test_x, tensor_test_y)
test_loader = DataLoader(test_set, batch_size=10, shuffle=True, num_workers=0)

print('start train')
model = Network()

loss_fn = nn.MSELoss()
optimizer = SGD(model.parameters(), lr=0.01)

def save_model():
    torch.save(model.state_dict(), './files/model.pth')

def testAccurary():
    model.eval()
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for data in test_loader:
            (images, labels) = data
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    accuracy = (100 * accuracy / total)
    return accuracy


def train(num_epochs):
    best_accuracy = 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            # print(labels)
            loss = loss_fn(outputs, labels.float())
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()
        
# we want to save the model if the accuracy is the best
save_model()

import matplotlib.pyplot as plt
import numpy as np

# Function to show the images
def imageshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Function to test the model with a batch of images and show the labels predictions
def testBatch():
    # get batch of images from the test DataLoader  
    images, labels = next(iter(test_loader))

    # show all images as one image grid
    imageshow(torchvision.utils.make_grid(images))
   
    # Show the real labels on the screen 
    print(classes)
    print(labels)
    print('Real labels: ', ' '.join('%5s' % classes[labels[j].index(1)] 
                               for j in range(10)))
  
    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)
    
    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)
    
    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] 
                              for j in range(10)))

# Let's build our model
train(5)
print('Finished Training')

# Test which classes performed well
# testModelAccuracy()

# Let's load the model we just created and test the accuracy per label
model = Network()
path = "./files/model.pth"
model.load_state_dict(torch.load(path))

# Test with batch of images
# testBatch()