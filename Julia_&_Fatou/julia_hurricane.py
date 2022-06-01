import os, sys
try:
    from pylab import *
except:
    os.system('pip install pylab')
    from pylab import *
try:
    from numpy import NaN
except:
    os.system('pip install numpy')
    from numpy import NAN
try:
    from matplotlib.colors import LogNorm
except:
    os.system('pip install matplotlib')
    from matplotlib.colors import LogNorm

# seed parameter from https://commons.wikimedia.org/wiki/File:Julia_set_%28red%29.png
s = -0.512511498387847167, 0.521295573094847167

res = .001    # grid resolution
max_it = 1000 # maximum number of iterations

figure(figsize = (10, 7))

def jul(a):
	z = a
	for n in range(1, max_it + 1):
		z = z**2 + s[0] + 1j * s[1]
		if abs(z) > 3:
			return n
	return NaN

X = arange(-2, 2 + res, res)
Y = arange(-1.5,  1.5 + res, res)
Z = zeros((len(Y), len(X)))

for iy, y in enumerate(Y):
	#print (iy + 1, "of", len(Y))
	for ix, x in enumerate(X):
		Z[-iy - 1, ix] = jul(x + 1j * y)

save("julia", Z)	# save array to file

imshow(Z, cmap = plt.cm.viridis, interpolation = 'none', norm = LogNorm(),
  extent = (X.min(), X.max(), Y.min(), Y.max()))
xlabel("Re(z)")
ylabel("Im(z)")
axis((-1.5, 1.5, -1, 1))
savefig("julia_python.svg")
show()