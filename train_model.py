'''
Trains model and saves to checkpoint 
'''
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import json
from PIL import Image
from torchvision import datasets, transforms, models
from workspace_utils import active_session
from load_data import load_data
from index_name import index_name

def train_model(data_dir, model_arch, hidden_units, learning_rate, epochs, gpu, save_dir):    
    
    device = torch.device("cuda:0" if (gpu) else "cpu")
    
    print (device)
    
    #Defines data loaders
    data_loader = load_data(data_dir)
    trainloader = data_loader['trainloader']
    validloader = data_loader['validloader']
    testloader = data_loader['testloader']
    
    #Initialize models
    densenet121 = models.densenet121(pretrained = True)
    vgg13 = models.vgg13(pretrained = True)

    model_archs = {'densenet121': densenet121, 'vgg13': vgg13}
    
    #Sets appropriate input units for the models
    if (model_arch=='densenet121'):
        input_units = 1024
    else:
        input_units = 25088
    
    #Assigns model specified in input
    model = model_archs[model_arch]    
    
    #Builds and trains network
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(input_units, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    
    with active_session():
        model.to(device)

        print_every = 4
        steps = 0

        for e in range(epochs):

            for inputs, labels in trainloader:
                running_loss = 0
                steps +=1

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0

                    model.eval()

                    with torch.no_grad():
                        for inputs, labels in validloader:
                            inputs, labels = inputs.to(device), labels.to(device)

                            logps = model.forward(inputs)
                            valid_loss = criterion(logps, labels)
                            ps = torch.exp(logps)

                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {e+1}/{epochs}  "
                          f"Train Loss: {running_loss/print_every:.3f}  "
                          f"Validation Loss: {valid_loss/len(validloader):.3f}  "
                          f"Validation Accuracy: {accuracy/len(validloader)*100:.3f}  ")

                    model.train()

    #Defines and saves checkpoint     
    save_path = save_dir + 'checkpoint.pth'    
    cat_to_name = index_name('cat_to_name.json')
    
    checkpoint = {'model_arch': model,
                  'input_size': input_units,
                  'output_size': 102,
                  'epochs': epochs,
                  'lr' : learning_rate,
                  'class_to_idx': cat_to_name,
                  'model_state_dict': model.state_dict(),
                  'optim_state_dict': optimizer.state_dict()}

    torch.save(checkpoint, save_path)
