import os, sys

from mandelbrot import mandelbrot, MAX_ITER

try:
	from PIL import Image, ImageDraw
except:
	os.system('pip install PIL')
	from PIL import Image, ImageDraw

try:
	import random
except:
	os.system('pip install random')
	import random

try:
	import numpy as np
except:
	os.system('pip install numpy')
	import numpy as np

#--------------------------------------
# Set up using full name as mandelbrot colors
#--------------------------------------
list_name = []

#######################################
# For raw_name, put YOUR name in quotes
#######################################
raw_name = 'Eric the Viking'
for letter in raw_name:
	number = ord(letter) - 96
	if number > 0:
	   list_name.append(number)
set_name = list(set(list_name))

WIDTH = 640 #1280 #2560 
HEIGHT = 480 #960 #1920

RE_START = -(random.choice(set_name))//26
RE_END   = (set_name[-1] - set_name[0]) //100
IM_START = -(np.mean(set_name)//150)
IM_END   = .75 
######################################
# Fun ways to modify the code for new results
######################################
#-------------------------------------
# IM_END, below are changes you can activate for differ results.
#   1. To comment out the previous line "IM_END = .75", 
#      simply put a hash mark (#) at the beginning of the line.

#   2. To active one of the other "IM_END = " lines,
#      simply remove the leading hash mark at the beginning of the line.

#IM_START = -(np.mean(set_name)//300)
#IM_START = -(np.mean(set_name)//100)
#IM_START = (set_name[0]/100) 
#IM_START = set_name[0] 
#--------------------------------------

palette = []




#(black and white) im = Image.new('RGB', (WIDTH, HEIGHT), (0, 0, 0))
im = Image.new('HSV', (WIDTH, HEIGHT), (0, 0, 0))
draw = ImageDraw.Draw(im)

for x in range(0, WIDTH):
	for y in range(0, HEIGHT):
		c = complex(RE_START + (x / WIDTH) * (RE_END - RE_START),
		    IM_START + (y / HEIGHT) * (IM_END - IM_START))
		m = mandelbrot(c)
		hue =255 - int(m * 255 / MAX_ITER *2) #Added "* 2" to MAX_ITER to make more intesting.
		saturation = 255
		if saturation > 255:
			saturation = 255
		value = (random.choice(set_name) * 100)
	######################################
        # Fun ways to modify the code for new results
        ######################################
		#--------------------------------------
		# Another variable to modify to get REALLY different results
		# Put a hash mark at beginning of the previous line:
		# value = (random.choice(set_name) * 100)
		# and remove the leading has mark from
		# the following line:
		# --> value = (random.choice(set_name) * 10)
		#---------------------------------------
		## value = (random.choice(set_name) * 10)

		if value > 250:
			value = 255
		draw.point([x, y], (hue, saturation, value))

im.convert('RGB').save('mandelbrot.png', 'PNG')
im.show()
