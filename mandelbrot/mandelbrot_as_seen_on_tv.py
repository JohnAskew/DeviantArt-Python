import os, sys
try:
	import numpy as np
except:
	os.system('pip install numpy')
	import numpy as np
try:
	from numba import jit
except:
	os.system('pip install numba')
	from numba import jit
try:
	import matplotlib as mpl
	import matplotlib.pyplot as plt
except:
	os.system('pip install matplotlib')
	import matplotlib as mpl
	import matplotlib.pyplot as plt

def mandelbrot(Re, Im, max_iter):
	c = complex(Re, Im)
	z = 0.0j

	for i in range(max_iter):
		z = pow(z,2) + c
		if (z.real*z.real + z.imag*z.imag) >= 4:
			return i

	return max_iter

columns = 2000
rows = 2000

result = np.zeros([rows, columns])
for row_index, Re in enumerate(np.linspace(-2, 1, num=rows)):
	for column_index, Im in enumerate(np.linspace(-1, 1, num=columns)):
		result[row_index, column_index] = mandelbrot(Re, Im, 100)

plt.figure(dpi = 100)
cmap = mpl.cm.gist_heat #afmhot #hot
#cmap_reversed = cmap.reversed()
plt.imshow(result.T, cmap=cmap, interpolation = 'bicubic', extent = [-2, 1, -1, 1])
plt.xlabel('Re')
plt.ylabel('Im')
plt.show()
