# Python code for Julia Fractal
import os, sys

try:
    from PIL import Image
except:
	os.system('pip install PIL')
	from PIL import Image

try:
	import math
except:
	os.system('pip install math')
	import math

# driver function
if __name__ == "__main__":
	
	# setting the width, height and zoom
	# of the image to be created
	w, h, zoom = 1920,1080,1

	# creating the new image in RGB mode
	bitmap = Image.new("RGB", (w, h), "antiquewhite")

	# Allocating the storage for the image and
	# loading the pixel data.c
	pix = bitmap.load()
	
	# setting up the variables according to
	# the equation to create the fractal
	#cX, cY = -0.7, 0.27015
	cX = -0.7
	#cY =  ((cX * math.sin(cX) -.1))#0.27015 # I like it - starfish
	
	#cY =  ((cX * math.tan(cX) -.1))#0.27015
	cY =  ((cX * math.sin(cX) -.1))#0.27015
	moveX, moveY = 0.0, 0.0
	#moveX, moveY = 1, -1 # only moves entire image down, so it's cut off.
	#maxIter = 255
	maxIter = 10100 # Nice purples and blues #505
	maxIter = 50500

	for x in range(w):
		for y in range(h):
			zx = 1.5*(x - w/2)/(0.5*zoom*w) + moveX
			zy = 1.0*(y - h/2)/(0.5*zoom*h) + moveY
			i = maxIter
			while zx*zx + zy*zy < 4 and i > 1:
				tmp = zx*zx - zy*zy + cX
				zy,zx = 2.0*zx*zy + cY, tmp
				i -= 1

			# convert byte to RGB (3 bytes), kinda
			# magic to get nice colors
			#pix[x,y] = (i << 21) + (i << 10) + i*8
			pix[x,y] = (i << 21) + (i << 10) + i*16

	# to display the created fractal
	bitmap.show()
