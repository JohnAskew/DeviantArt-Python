import os, sys

try:
    from PyAstronomy import pyasl
except:
    os.system('pip install PyAstronomy')
    from PyAstronomy import pyasl

T, mu, g = 290, 28.97, 9.8
she = pyasl.atmosphericScaleHeight(T, mu, g)

print("Earth")
print(f"T, mu, g = {T} K, {mu}, {g} m/s**2")
print(f"Scale height = {she:4.1f} [km]")


T, mu, mp, rp = 165, 2.2, 1, 1
shj = pyasl.atmosphericScaleHeight_MR(T, mu, mp, rp, "J")

print("Jupiter")
print(f"T, mu, mp, rp = {T} K, {mu}, {mp} [MJ], {rp} [RJ]")
print(f"Scale height = {shj:4.1f} [km]")