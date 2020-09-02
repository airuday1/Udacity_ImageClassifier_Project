'''
Successfully reads in an image and a checkpoint then prints the most likely image class and it's associated probability

--image_path : Flag for image path
--checkpoint : Flag for checkpoint location
--top_k : Flag for setting top_k
--gpu : Flag for enabling gpu. (Default: False)
--category_names : Flag to specify json file for category mapping
'''
from inference_input_args import inference_input_args
from predict_image import predict_image
from load_checkpoint import load_checkpoint
from index_name import index_name

def main():
    #Parse command line inputs
    in_args = inference_input_args()
    image_path = in_args.image_path
    checkpoint = in_args.checkpoint
    top_k = in_args.top_k
    gpu = in_args.gpu
    
    #Loads a JSON file that maps the class values to other category names
    cat_to_name = index_name(in_args.category_names)

    
    #Load checkpoint
    model = load_checkpoint(checkpoint) 
    
    #To retrieve top probabilities and classes
    top_p, top_class = predict_image(image_path, model, top_k, gpu)

    flower_names = []
    
    #Print top probabilities and classes
    for i in top_class:
        for key in i:
            flower_names.append(cat_to_name[str(key.tolist())])
    
    print ('Top Class Probabilities : ', top_p)
    print ('Top Class Categories :', flower_names)

# Call to main function to run the program
if __name__ == "__main__":
    main()    
    