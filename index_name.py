'''
Loads json file to dictionary
'''
import json

def index_name(filepath):
    
    #Load json file to cat_to_name dictionary
    with open(filepath, 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name