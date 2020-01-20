from dataset import *
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch.optim as optim


from utilities import helper
from utilities import train_class
from utilities import test_class

from model import CNN_Classifier_1, CNN_Classifier_2

# Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor()])

test_transforms = transforms.Compose([ transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor()])


data_dir = 'data/Cat_Dog_data'

train_data = CatDogDataset(data_dir + '/train', transform=train_transforms)
test_data = CatDogDataset(data_dir + '/test', transform=test_transforms)

validation_split = .2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(train_data)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)


train_loader = torch.utils.data.DataLoader(train_data, batch_size=32,
                                             sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=32, 
                                             sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)


#############
import torchvision
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('try/dog_vs_cat')

# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# create grid of images
img_grid = torchvision.utils.make_grid(images)

# write to tensorboard
writer.add_image('10_dog_cat_images', img_grid)

##############

if __name__=='__main__':
    model_scratch = CNN_Classifier_2(n_feature=16, output_size=2)

    writer.add_graph(model_scratch, images)
    writer.close()

    loaders_scratch = {'train': train_loader, 'valid' : valid_loader, 'test': test_loader}

    ### select loss function
    criterion_scratch = nn.CrossEntropyLoss()

    ### select optimizer
    optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=0.03)

    # move tensors to GPU if CUDA is available
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        print("Your code is running on GPU :)\n")
        model_scratch.cuda()
    else:
        print("Your code is running on CPU :)\n")
    print(model_scratch)

    # Training
    print('Training ....')
    model_scratch = train_class.train(5, loaders_scratch, model_scratch, optimizer_scratch, 
                      criterion_scratch, use_cuda, writer, './model/model_cnn_2.pt') # You can rename this file to save different check point

    # load the model that got the best validation accuracy
    # model_scratch.load_state_dict(torch.load('./model/model_scratch.pt')) # uncomment only to load the saved model

    # Testing
    print('Testing ....')
    test_class.test(loaders_scratch, model_scratch, criterion_scratch, use_cuda)