# Notes

- write down which symbols to take out so can add them in later stages
- size of input image for the net is hard coded in code/generate_imgs.py img_to_array()
- removed \div and - from operators: opencv not recognizing any contour

# Problems

- different font sizes cv2.findcontour messes up on smaller fonts 
(chose 35 fontsize for now)
- some symbols are very similar
- some symbols have more than one rectangle
- "-" does not have a rectangle

# TODO

- add noise to images. will help with when other symbols get into cv2 border
- train model and optimize tuning parameters (max_epochs, learning_rate, momentum)
- create algorithm to convert to latex
- create a flask webapp with bootstrap

# Quick Problems


# Files

code/generate_images.py

- generate_images: input a list of values, output a latex image
- get_img: crops an image from the file
- img_to_array: converts all images in folder to a 28x28 matrix and saves them
- compile_images: compiles a folder of jsons to one json
- add_noise: add noise to an image

code/model.py

imgs/: contains subfolders of images for every symbol

data/images/: contains json for each subfolder of imgs and a compiled json

data/*.csv: each file contains a list of symbols to convert to latex image
