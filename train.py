'''
Main Function - Successfully trains a new network on a dataset of images and saves checkpoint

--gpu : Flag to enable gpu. Default: False
--arch : Flag to set model architecture. Options : densenet121 / vgg13
--data_dir : Flag to specify path to folder of image datasets 
--save_dir : Flag to specify location to save checkpoint
--epochs : Flag to set # epochs
--learning_rate : Flag to set lr
--hidden_units : Flag to set # hidden units 

'''
from get_input_args import get_input_args
from train_model import train_model

def main():
    in_arg = get_input_args()
    
    data_dir = in_arg.data_dir
    model_arch = in_arg.arch
    hidden_units = in_arg.hidden_units
    learning_rate = in_arg.learning_rate
    epochs = in_arg.epochs
    gpu = in_arg.gpu
    save_dir = in_arg.save_dir
    
    train_model(data_dir, model_arch, hidden_units, learning_rate, epochs, gpu, save_dir)

# Call to main function to run the program
if __name__ == "__main__":
    main()