import os, sys
from mandelbrot_info import mandelbrot, MAX_ITER
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

raw_name = 'elizabeth smith'
for letter in raw_name:
	number = ord(letter) - 96
	if number > 0:
	   list_name.append(number)
set_name = list(set(list_name))

WIDTH = 1280 #2560 
HEIGHT = 960 #1920

RE_START = -(random.choice(set_name))//26#-2
RE_END   = (set_name[-1] - set_name[0]) //100 
IM_START = -(np.mean(set_name)//300) 
IM_END   = .75
palette = []

im = Image.new('HSV', (WIDTH, HEIGHT), (0, 0, 0))
draw = ImageDraw.Draw(im)

for x in range(0, WIDTH):
	for y in range(0, HEIGHT):
		c = complex(RE_START + (x / WIDTH) * (RE_END - RE_START),
		    IM_START + (y / HEIGHT) * (IM_END - IM_START))
		m = mandelbrot(c)
		hue =255 - int(m * 255 / MAX_ITER)
		saturation = 255
		if saturation > 255:
			saturation = 255
		value = (random.choice(set_name) * 10)
		if value > 250:
			value = 255
		draw.point([x, y], (hue, saturation, value))

im.convert('RGB').save('mandelbrot.png', 'PNG')
