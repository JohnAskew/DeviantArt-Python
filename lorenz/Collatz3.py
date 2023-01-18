import os, sys
from itertools import cycle
from matplotlib import pyplot as plt

COLORS = cycle( 
    [
        "#372248", # light purple
        "#110099", #"#5B85AA", # purple-blue
        "#8800ff", #"#414770", # navy blue
        "#087f04", #Plant green
        "Yellow" # "#40C9A2"  # green
    ]
    )

PATTERN_SHIFT = [
    (1, 0),
    (1.1, -5),
    (0.89, -2),
    (0.75, -0.5),
    (0.95, -20)
]

collatz = []

for num in range(1, 50000):
    length = 1
    while not num == 1:
        if num %2 == 0:
            num /= 2
        else:
            num = 3 * num + 1
        length +=1
    collatz.append(length)

def shift_collatz(seq, m, c):
    return [m*x + c for x in seq]

fig = plt.figure(figsize=(10, 6), dpi=150)
fig.patch.set_facecolor("#f8ffff")
ax = plt.subplot()
ax.set_xlim(10000,40000)
ax.set_ylim(1, 300) #250) #150)
ax.set_axis_off()

for pattern in PATTERN_SHIFT:
    ax.plot(shift_collatz(collatz, *pattern),
    "H",
    markersize = 10, #20,
    color = next(COLORS),
    alpha = 0.03) #0.03)

plt.show()