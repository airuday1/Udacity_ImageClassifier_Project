'''
Runs image through pre-trained model and returns top probabilities and their corresponding classes
'''
import torch
from torchvision import models
from PIL import Image
from process_image import process_image

def predict_image(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Code to predict the top probabilities and classes from an image file
    device = torch.device("cuda:0" if (gpu) else "cpu")
    
    print (device)
    
    image = process_image(image_path)
               
    image.unsqueeze_(0)
    
    model.to(device)
        
    model.eval()
            
    with torch.no_grad():       
        
        logps = model.forward(image.to(device)) 
        ps = torch.exp(logps)

        top_p, top_class = ps.topk(topk, dim=1)           

    return top_p, top_class