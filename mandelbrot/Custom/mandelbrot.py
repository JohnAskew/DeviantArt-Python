import os, sys
try:
	from PIL import Image, ImageDraw
except:
	os.system('pip install PIL')
	from PIL import Image, ImageDraw

#--------------------------------------
# Global declarations
#--------------------------------------
MAX_ITER = 100# 20  #80
#--------------------------------------
def mandelbrot(c):
#--------------------------------------
	z = 0
	n = 1
	#while abs(z) <=2 and n < MAX_ITER:
	while abs(z) <=5 and n < MAX_ITER:
		#z = z **2 + c
		z = pow(z,3) + c
		n += 1
	return n

#--------------------------------------
# MAIN LOGIC
#--------------------------------------
if __name__ == "__main__":
	for a in range(-10, 10, 5):
		for b in range(-10, 10, 5):
			c = complex(a/10, b/10)
			print(c, mandelbrot(c))
