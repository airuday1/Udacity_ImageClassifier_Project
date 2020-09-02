'''
Loads training, test and validation datasets from specified directory
'''
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from get_input_args import get_input_args

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #Defines transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    #Loads the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    #Defines the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 64)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64)
    
    return {'trainloader': trainloader, 'testloader': testloader, 'validloader': validloader}
