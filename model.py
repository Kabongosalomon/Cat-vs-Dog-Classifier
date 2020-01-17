# Attempt to build a network to classify cats vs dogs from this dataset
from torch import nn, optim
import torch.nn.functional as F


class CNN_Classifier_1(nn.Module):
    def __init__(self, n_feature, output_size):
        super(CNN_Classifier_1, self).__init__()
        self.n_feature = n_feature
        
        ## Define layers of a CNN
        self.conv1 = nn.Conv2d(3, n_feature*2**1, 3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(n_feature*2**1, n_feature*2**2, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(n_feature*2**2, n_feature*2**3, 3, padding=1)
        
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128*7*7, 500)
        self.fc2 = nn.Linear(500, output_size)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, verbose=False):
        ## Define forward behavior~
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # flatten image input
        x = x.view(-1, 128*7*7)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)

        return x

class CNN_Classifier_2(nn.Module):
    def __init__(self, n_feature, output_size):
        super(CNN_Classifier_2, self).__init__()
        self.n_feature = n_feature
        
        ## Define layers of a CNN
        self.conv1 = nn.Conv2d(3, n_feature*2**1, 3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(n_feature*2**1, n_feature*2**2, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(n_feature*2**2, n_feature*2**3, 3, padding=1)
        self.conv4 = nn.Conv2d(n_feature*2**3, n_feature*2**4, 3, padding=1, stride=2)
        
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256*2*2, 500) # Computer this automatically using scrept in the dog_breed classifier notebook
        self.fc2 = nn.Linear(500, output_size)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, verbose=False):
        ## Define forward behavior~
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        # flatten image input
        x = x.view(-1, 256*2*2)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)

        return x