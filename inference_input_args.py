'''
Code to parse command line inputs
'''
import argparse

def inference_input_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image_path', type = str, help = 'Specify image path')
    
    parser.add_argument('--checkpoint', type = str, default = 'checkpoint.pth', help = 'Specify checkpoint.pth path')
    
    parser.add_argument('--top_k', type = int, default = 3, help = 'Specify the # of top class predictions')
    
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'File for mapping categories to real names')
    
    parser.add_argument('--gpu', type = bool, default = False, help = 'Switch to True to use GPU')
    
    return parser.parse_args()