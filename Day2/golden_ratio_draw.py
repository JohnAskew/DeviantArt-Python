#!/usr/bin/env python #Mac users may need to change "python" to "python3".
import os, sys
'''
Python program for Plotting Fibonacci
 spiral fractal using Turtle
--> To make this an interactive program: comment out the variable "n" 
and uncomment "#n = int(input('Enter the number of iterations (must be > 1): '))".
Now, run in Terminal window and choose the number of fibonacci iterations.
* Your terminal code to start this program in the terminal window is either:
    a) python golden_ratio_draw.py
    b) python3 golden_ratio_draw.py

For 8.5x11 size figure, answer "iteration" question with 7.
'''
try:
    import turtle
except:
    os.system('pip install turtle')
    import turtle

try:
    import math
except:
    os.sysem('pip install math')
    import math


#-----------------------------------
def fiboPlot(n):
#-----------------------------------
    a = 0
    b = 10
    square_a = a
    square_b = b

    # Setting the colour of the plotting pen to blue
    x.pencolor("blue")

    # Drawing the first square
    #x.forward(b * factor)
    x.forward(b* factor)
    x.left(90)
    x.forward(b * factor)
    x.left(90)
    x.forward(b * factor)
    x.left(90)
    x.forward(b * factor)

    # Proceeding in the Fibonacci Series
    temp = square_b
    square_b = square_b + square_a
    square_a = temp
    
    # Drawing the rest of the squares
    for i in range(1, n):
        x.backward(square_a * factor)
        x.right(90)
        x.forward(square_b * factor)
        x.left(90)
        x.forward(square_b * factor)
        x.left(90)
        x.forward(square_b * factor)

        # Proceeding in the Fibonacci Series
        temp = square_b
        square_b = square_b + square_a
        square_a = temp

    # Bringing the pen to starting point of the spiral plot
    x.penup()
    x.setposition(factor, 0)
    x.seth(0)
    x.pendown()

    # Setting the colour of the plotting pen to red
    x.pencolor("red")

    # Fibonacci Spiral Plot
    x.left(90)
    for i in range(n):
        print(b)
        fdwd = math.pi * b * factor / 2
        fdwd /= 90
        for j in range(90):
            x.forward(fdwd)
            x.left(1)
        temp = a
        a = b
        b = temp + b

############################################
# M A I N   L O G I C   S T A R T S   H E R E
############################################

# Here, 'factor' signifies the multiplicative
# factor which expands or shrinks the scale
# of the plot by a certain factor.
factor = 5
height = 1300
width = 1050

# Taking Input for the number of
# Iterations our Algorithm will run
#n = int(input('Enter the number of iterations (must be > 1): '))
n = 5
# Plotting the Fibonacci Spiral Fractal
# and printing the corresponding Fibonacci Number
if n > 0:
    print("Fibonacci series for", n, "elements :")
    x = turtle.Turtle()
    x.speed(0)
    screen = turtle.Screen()
    screen.setup(height, width)
    screen.title("Golden Ratio as a Fibonacci Spiral visual.")
    fiboPlot(n)
    canvas = screen.getcanvas()
    turtle.getscreen().getcanvas().postscript(file=r"golden_ratio.eps")
    turtle.done()
else:
    print("Number of iterations must be > 0")

  
