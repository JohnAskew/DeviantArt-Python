import os, sys
try:
    import numpy as np
except:
    os.system('pip install numpy')
    import numpy as np
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mplc
except:
    os.system('pip import matplotlib')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mplc

#def lorenz(x, y, z, s=20, r=50, b=5.000):    
#--------------------------------------
def lorenz(x, y, z, s=10, r=28, b=2.667):
#--------------------------------------

    """
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    """
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot


dt = 0.01

num_steps = 10000
#num_steps = 50000
rgb_cycle = []

# Need one more for the initial values
xs = np.empty(num_steps + 1)
ys = np.empty(num_steps + 1)
zs = np.empty(num_steps + 1)

# Set initial values
xs[0], ys[0], zs[0] = (0., 1., 1.05)


# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps):
    x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
    rgb_cycle.append(((abs(x_dot )%255/255), (abs(y_dot)%255/255), (abs(z_dot)%255/255)))
    #print(f'rgb_cycle={rgb_cycle[i]}')
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)
    
# Plot
ax = plt.figure(figsize=(12,10)).add_subplot(projection='3d')
for i in range(num_steps):
    ax.plot(xs[i], ys[i], zs[i], c=mplc.to_rgb(rgb_cycle[i]), 
    # color='navy', 
    linestyle='-', 
    marker='o', 
    markersize=.5,
    linewidth=.2)

#ax.set_title("Lorenz Attractor")
plt.suptitle("Illustrating that a butterfly flapping its wings in Brazil\ncould, theoretically, influence the formation of a tornado in Georgia,\nThe 'butterfly effect',\na concept within chaos theory.\ndescribes how a small change in initial conditions\ncan lead to drastically different outcomes in a complex system\n THAT IS, until System Complexity crosses into 'Large Numbers', in which Chaos becomes deterministic!\n - Zion Logbo...I mean, duh,\nlike forget Boyle's law, it's all about PATTERN metamorphism,\nyou know, the sheer weight of Large Numbers\nwill deterministically collapse the wave function, despite never being formally observed...\nHeisenberg and Schroedinger are idiots,\nwith their ivory-tower theories about\nrelative maximas manifesting themselves as absolute states, truth-tables,\nand job-justifying propaganda on why ''they'' are the only one who can solve math problems.\nThey could not even be bothered to articulate what formally constitutes an 'obervation',\nmuch less whether tachyons merit or simply posture as observation criteria...\nAND just where are your irreproducable definitions of obervations recorded, on a cave wall?\nYou bunch of quantum-mechanic troglodytes' ")
ax.axis('off')
#ax.view_init(15,65)
#plt.savefig("lorenz_attractor.png", bbox_inches='tight')
plt.savefig("butterfly_effect.png", bbox_inches='tight')

plt.show()