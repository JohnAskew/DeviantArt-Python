from __future__ import print_function, division
import os, sys

try:
    import numpy as np
except:
    os.system('pip install numpy')
    import numpy as np

try:
    from PyAstronomy import pyasl
except:
    os.system('pip install PyAstronomy')
    from PyAstronomy import pyasl

try:
    import datetime
except:
    os.system('pip install datetime')
    import datetime

# Convert calendar date into JD
# use the datetime package
jd = datetime.datetime.now() #(2023, 7, 16)
jd = pyasl.jdcnv(jd)
print("Current datetime in Julian: JD = " + str(jd))
pos = pyasl.sunpos(jd, full_output=True)
print("Coordinates of the Sun (ra, dec): %g, %g" % (pos[1][0], pos[2][0]))
print("Solar ecliptic longitude = %g and obliquity = %g" % (pos[3][0], pos[4][0]))

# Get the Sun's RA and DEC values for a period of time.
startjd = datetime.datetime(2023, 7, 16)
endjd = datetime.datetime(2025, 7, 17)
# Convert into Julian dates
startjd = pyasl.jdcnv(startjd)
endjd = pyasl.jdcnv(endjd)
print()
pos = pyasl.sunpos(startjd, end_jd=endjd, jd_steps=20, plot=False, full_output=True)

for i in range(len(pos[0])):
    print("At JD = %.2f: Sunrise ascent = %g, Sun Decline = %g" % (pos[0][i], pos[1][i], pos[2][i]))