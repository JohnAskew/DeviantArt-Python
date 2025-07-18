import os, sys

try:
    import turtle
except:
    os.system('pip install turtle')
    import turtle

try:
    import math
except:
    os.system('pip install math')
    import math

try:    
    import random
except:
    os.system('pip install random')
    import random

#############################
# If you do not see 8 planets when turtle (app) completes,
#    then setting "scale" to a larger number will shrink the 
#    size of the planets and solar system.
#############################
# quantity of miles to scale down universe by
scale = 125000 #75000

# make the "space" black
screen = turtle.Screen()
screen.setup(width=1280, height=960, startx=100, starty=50)
screen.title("Kepler's scaled view of our Solar System.")


screen.bgcolor("black")
# Set the colormode to 255 to use RGB integer values (0-255)
screen.colormode(255)

# generate sun turtle and move it to the center
sun = turtle.Turtle()
sun.setposition(0,0)
sun.penup()


# set the sun's color, and get ready to fill the shape with the color
sun.color(255, 204, 51)
sun.pendown()
sun.begin_fill()
# set a planetary scale to make the planets larger
planetScale = 60 #100

# calculate the rescaled sun's radius
sunRadius = 432690 / scale

# circumference is 2 * pi * radius
sunCircumference = 2 * math.pi * sunRadius

# draw a circle to represent the sun–divide circumference by 36 to move forward incrementially
# each line of the circle builds into a full circle–by turning by a small degree each time
for i in range(36):
  sun.forward(sunCircumference / 36)
  sun.right(10)

# finish drawing the sun, so fill its color
sun.end_fill()

# hide the sun's turtle after done drawing
sun.hideturtle()

# generate planet turtle to draw our planet
planet = turtle.Turtle()
planet.speed(0)

## Draw the planetary orbit; uncomment lines below to draw orbits
# orbit = turtle.Turtle()
# orbit.color("white")

# Draw a planet given a list of colors [r,g,b], the distance from the sun, and the radius of the planet
def drawPlanet(color, distance, radius):
  # scale the given radius by a planet scaler to make it larger, then divide it by the largest scale
  radius = radius * planetScale / scale
  # calculate circumference given new radius 
  circumference = 2 * math.pi * radius
  print(circumference)
  
  # add up all radii and scaled distances to calculate scaled distance from the sun, then divide by a new scale to make it appear on the screen
  distFromSun = sunRadius + ((radius + distance) / scale) / 50
  
  ## Draw the planetary orbit; uncomment lines below and above to draw orbits
  # orbit.penup()
  # orbit.setposition(0,0)
  # orbit.showturtle()
  
  # orbit.setheading(0)
  # orbit.forward(distFromSun)
  # orbit.setheading(270)
  # orbitCircumference = 2 * math.pi * distFromSun
  
  # orbit.pendown()
  # for i in range(36):
  #   orbit.forward(orbitCircumference / 36)
  #   orbit.right(10)
    
  # orbit.hideturtle()
  
  # extract rgb variables from passed color list
  r = color[0]
  g = color[1]
  b = color[2]
  
  # rest planet to start position
  planet.setposition(0,0)

  # set the planet to move in a random direction
  planet.setheading(random.randint(0,360))

  # make planet visible, set color, penup to get ready to move
  planet.showturtle()
  planet.color(r,g,b)
  planet.penup()
  
  # move planet forward to correct position
  planet.forward(distFromSun)

  # get ready to draw and fill a shape
  planet.pendown()
  planet.begin_fill()

  # draw a circle based on calculated circumference
  for i in range(36):
    planet.forward(circumference / 36)
    planet.right(10)
  
  # stop the fill into the planet, ending it with a full shape
  planet.end_fill()

  # hide the turtle again and stop drawing
  planet.hideturtle()
  planet.penup()

## Draw all the planets! ##
# draw mercury
drawPlanet([151, 151, 159], 35980000, 1516)
  
# venus
drawPlanet([211, 156, 126], 67240000, 3760.4)

# earth
drawPlanet([140, 177, 222], 92950000, 3958.8)

# mars
drawPlanet([198, 123, 92], 141600000, 2106.1)

# jupiter
drawPlanet([211, 156, 126], 483800000, 43441)

# saturn
drawPlanet([197, 171, 110], 890000000, 36184)

# uranus
drawPlanet([213, 251, 252], 1784000000, 15759)

# neptune
drawPlanet([62, 84, 232], 2793000000, 15299)

turtle.done()

### BACKGROUND INFORMATION – COLOR, DISTANCE, RADIUS ###
### RGB Codes ###
# Sun: (255, 204, 51) - [255, 204, 51]
# Mercury: (151, 151, 159) - [151, 151, 159]
# Venus: (211, 156, 126) - [211, 156, 126]
# Earth: (140, 177, 222) - [140, 177, 222]
# Mars: (198, 123, 92) - [198, 123, 92]
# Jupiter: (211, 156, 126) - [211, 156, 126]
# Saturn: (197, 171, 110) - [197, 171, 110]
# Uranus: (213, 251, 252) - [213, 251, 252]
# Neptune: (62, 84, 232) - [62, 84, 232]

### Planetary Distance from Sun (miles) ###
# Sun: 0 mi
# Mercury: 35.98 million mi
# Venus: 67.24 million mi
# Earth: 92.96 million mi
# Mars: 141.6 million mi
# Jupiter: 483.8 million mi
# Saturn: 890.9 million mi
# Uranus: 1.784 billion mi
# Neptune: 2.793 billion mi

### Planetary Radii (miles) ###
# Sun: 432690 mi
# Mercury: 1516 mi
# Venus: 3760.4 mi
# Earth: 3958.8 mi
# Mars: 2106.1 mi
# Jupiter: 43441 mi
# Saturn: 36184 mi
# Uranus: 15759 mi
# Neptune: 15299 mi