'''
Pre-processes input image to run in model
'''
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model
    '''
    
    # Processes a PIL image for use in a PyTorch model
    img_pil = Image.open(image)
    
    
    pil_transforms = transforms.Compose([
                                         transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    
        
    return (pil_transforms(img_pil))