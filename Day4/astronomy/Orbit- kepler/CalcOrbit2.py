#  https://pyastronomy.readthedocs.io/en/latest/pyaslDoc/aslDoc/keplerOrbit.html
# 3rd script at bottom 
# Just plugged the numbers in- something works, now I have to try and understand it. Monday 27, March, 2023

# Example: Calculate binary orbits
import numpy as np
import matplotlib.pylab as plt
from PyAstronomy import pyasl

m2m1 = 0.3
tau = 12.5

bo = pyasl.BinaryOrbit(m2m1, 2.3, 17., e=0.5, tau=tau, Omega=180, w=0., i=0.)

ke1 = bo.getKeplerEllipse_primary()
ke2 = bo.getKeplerEllipse_secondary()

# Input time in seconds
t = np.linspace(10, 10+15, 35) * 86400

r1, r2 = bo.xyzPos(t)
v1, v2 = bo.xyzVel(t)

plt.subplot(2,1,1)
plt.plot(r1[::,0], r1[::,1], 'b.-', label="Primary orbit")
plt.plot(r2[::,0], r2[::,1], 'r.-', label="Secondary orbit")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.legend()
# Relative distance of masses (centers)
rd = np.sqrt(np.sum((r1-r2)**2, axis=1))
plt.subplot(2,1,2)
plt.plot(t/86400, rd, 'b.-', label="Relative distance")
plt.axvline(tau, ls=':', c='k', label="Time of periastron")
plt.xlabel("Time [days]")
plt.ylabel("Distance [m]")
plt.legend()
plt.show()

plt.subplot(3,1,1)
plt.plot(t/86400, v1[::,0]/1e3, 'b.-', label="Primary")
plt.plot(t/86400, v2[::,0]/1e3, 'r.-', label="Secondary")
plt.xlabel("Time [days]")
plt.ylabel("vx [km/s]")
plt.legend()
plt.subplot(3,1,2)
plt.plot(t/86400, v1[::,1]/1e3, 'b.-', label="Primary")
plt.plot(t/86400, v2[::,1]/1e3, 'r.-', label="Secondary")
plt.xlabel("Time [days]")
plt.ylabel("vy [km/s]")
plt.legend()
# Orbit velocities
plt.subplot(3,1,3)
ov1 = np.sqrt(np.sum(v1**2, axis=1))
ov2 = np.sqrt(np.sum(v2**2, axis=1))
plt.plot(t/86400, ov1/1e3, 'b.-', label="Primary")
plt.plot(t/86400, ov2/1e3, 'r.-', label="Secondary")
plt.xlabel("Time [days]")
plt.ylabel("Orbit velocity [km/s]")
plt.legend()
plt.show()