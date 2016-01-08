# Notes

- write down which symbols to take out so can add them in later stages
- size of input image for the net is hard coded in code/generate_imgs.py img_to_array() and latex_to_code.py predict_symbol()
- removed \div from operators: opencv not recognizing any contour

# Problems

- different font sizes cv2.findcontour messes up on smaller fonts 
(chose 35 fontsize for now)
- some symbols are very similar
- "-" does not have a rectangle

# TODO

- train model and optimize tuning parameters (max_epochs, learning_rate, momentum)
- create algorithm to convert to latex
    - handle: \int, \frac, super/sub script
    - create functions for all of these

- generate latex algorithm: go left to right. if "-" then check if is \frac by comparing if there is anything above and below

- add buffer to images that are small in either dimension

# Quick Problems

# Files

code/generate_images.py

- generate_images: input a list of values, output a latex image
- get_img: crops an image from the file
- img_to_array: converts all images in folder to a 28x28 matrix and saves them
- compile_images: compiles a folder of jsons to one json
- add_noise: add noise to an image

code/model.py

code/latex_to_code.py

- find_symbols: input a b/w image and output a list of bounding rectangles containg all the symbols in the image.
- predict_symbol: used by generate latex(). predicts which symbol is contained in the rectangle.
- generate_latex: takes input of all symbols and arranges them to latex code.

imgs/: contains subfolders of images for every symbol

data/images/: contains json for each subfolder of imgs and a compiled json

data/*.csv: each file contains a list of symbols to convert to latex image

