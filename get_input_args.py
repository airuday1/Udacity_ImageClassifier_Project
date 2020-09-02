'''
Code to parse command line inputs
'''
import argparse

def get_input_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type = str, default = 'flowers/', help = 'Path to folder of image datasets')
    
    parser.add_argument('--save_dir', type = str, default = '', help = 'Path to save checkpoint')
    
    parser.add_argument('--arch', type = str, default = 'densenet121', help = 'Specify CNN model architecture')
    
    parser.add_argument('--epochs', type = int , default = 2, help = 'Define the number of epochs')
    
    parser.add_argument('--learning_rate', type = float, default = 0.003, help = 'Define the learning rate')
    
    parser.add_argument('--hidden_units', type = int, default = 512, help = 'Define the number of hidden units')
    
    parser.add_argument('--gpu', type = bool, default = False, help = 'Switch to True to use GPU')
        
    return parser.parse_args()