# Latex Image to Code

Converts a mathematical image to latex code. First, it finds each symbol
in the image and uses a convolutional neural network to determine what
symbol it is. Once each symbol is identified, it puts back together the
latex code based on positions and sizes.

# Package Dependencies

- cv2
- nolearn/lasagne
- numpy
- pandas
- ipython
- matplotlib
- cPickle
- Flask

# Files

- code/generate_images.py: Contains functions to generate every symbol.
Used for training the neural net for symbol recognition.
- code/model.py: The neural network used for symbol recognition.
- code/Latex2Code.py: The class to put together the symbols to a latex
code string.
- webapp/: Folder containing code to launch a Flask webapp.

# Examples

Input 1:

![alt text](img/example.png)

Output 1: "\frac{\frac{A}{\beta}+\gamma}{x+y}=z"

Input 2:

![alt text](img/example2.png)

Output 2: "yA_{Ay}^{xy}"

# Results

The test images were randomly generated with a script that allowed for 
adjusting the complexity of the equation. A more complex equation would 
have more super/subscripts, fractions, and symbols.

Achieved 83% accuracy on equations with two operators and 35% on equations 
with 3-4 operators, where an operator is the standard +, -, * or others 
such as =, \frac, super/subscript, etc. An output is considered correct only 
if the output exactly matches the input that made the Latex code. 
The equations with two operators had an average of 7 symbols, and the 
equations with 3-4 operators had an average of 15 symbols.

