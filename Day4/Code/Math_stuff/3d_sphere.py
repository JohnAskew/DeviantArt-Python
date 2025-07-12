#!/usr/bin/env python
import os, sys

try:
    import matplotlib.pyplot as plt
except:
    os.system('pip install matplotlib')
    import matplotlib.pyplot as plt

try:
    import numpy as np
except:
    os.system('pip install numpy')
    import numpy as np
#----------------------------------
# HOUSEKEEPING - set up processing inputs
#----------------------------------
plt.rcParams['figure.figsize'] = (12,10)
plt.rcParams['axes.facecolor'] = 'darkblue'
plt.rcParams['axes.grid'] = True

#-----------------------------------
# Make data
#-----------------------------------
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 10 * np.outer(np.cos(u), np.sin(v))
y = 10 * np.outer(np.sin(u), np.sin(v))
z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

#------------------------------------
# BEGIN Main Logic
#------------------------------------

fig = plt.figure()
fig.patch.set_facecolor('#f8ffff') ##00008C')

plt.axis(False)
plt.grid(True, color='#f8ffff')
plt.title('Planes on sphere and x,y,z axis box - For Drawing Class Day3.')


ax = plt.axes(projection='3d') # 'hammer') #3d')
ax.plot_surface(x, y, z, color='#FF00FF', linestyle = '--', linewidth=5, alpha=1)
#ax.grid(True, color='black', lw=50)

#ax = fig.add_subplot(projection='3d')


# Plot the surface
plt.show()