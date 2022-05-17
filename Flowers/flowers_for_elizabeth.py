""" This pgm produces a star with a round center """
import os, sys
try:
	import turtle as tt
except:
	os.system('pip install turtle')
	import turtle as tt
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
try:
	from  colorsys import *
except:
	os.system("pip install colorsys")
	from colorsys import *

#--------------------------------------
# Set up and Housekeeping
#--------------------------------------
colors_flowers_yellow = '#FFD700'
colors_flowers_blue   = "blue" 
center_color = [(colors_flowers_yellow), colors_flowers_blue, (118, 238, 0)]
colors_rose = [(255, 155, 230), colors_flowers_yellow, "red", "violet"]
colors_carnation = ["pink", colors_flowers_yellow, "red", colors_flowers_blue]
colors_swirl = ["red", colors_flowers_yellow, "pink", "red", "pink"]
colors_swirl2 = ["violet", colors_flowers_yellow, "pink", "violet", "pink"]
colors_swirl3 = [colors_flowers_blue, colors_flowers_yellow, "violet", "violet", colors_flowers_blue]
colors_swirl4 = [(0,128,128), (189,252,201), (72,209,204), (245,255,250), (192,255,62) ] 
carnation_multicolor_colors = [colors_flowers_yellow, "pink", "red", "green", colors_flowers_blue, "violet", "indigo"]
colors_arc=['red', 'pink', 'blue',]
colors_spiral = ["red", "green", "magenta", colors_flowers_blue, "purple", "orange"]



star5_rng = 50 #default. Will be overridden in call statement
#--------------------------------------
# Set up Turtle and "star"
#--------------------------------------
tt.hideturtle()
star = tt.Turtle()
star.hideturtle()
tt.screensize(canvwidth=1280, canvheight=960) #, bg="skyblue")
win = tt.Screen()
win.bgpic("Elizabeth_bkgrnd_classic_blue.png")
star.speed("fastest")
tt.up()
tt.tracer(0)
tt.setposition(0, -300)
tt.colormode(255)

#######################################
# FUNCTIONS
#######################################
#--------------------------------------
def flower_spiral(rng, ang, factor, x, y):
#--------------------------------------
    star.up()
    star.goto(x, y)
    star.down()
    for i in range(rng):
        star.pencolor(colors_spiral[i%len(colors_spiral)])
        star.width(i/factor + 1)
        star.forward(i)
        star.left(ang)
    star.up()
    tt.setposition(0, -300)
#--------------------------------------
def arcFlower(rng, petals, x, y):
#--------------------------------------
	tt.up()
	tt.goto(x, y)
	tt.down()
	for i in range(rng): 
		tt.begin_fill()
		for j in range(petals):
			tt.fd(i/10) 
			tt.lt(32)
			tt.heading()
		tt.fillcolor(colors_arc[i%len(colors_arc)])
		tt.end_fill()
		tt.left(198)

#--------------------------------------
def flower_carnation_multi(cm_range, cm_angle, cm_radius, x, y):
#--------------------------------------
	tt.up()
	tt.setpos(x, y)
	tt.pensize(2)
	h=0.5
	tt.bgcolor("black")

	for i in range(cm_range):
		tt.pencolor(carnation_multicolor_colors[cm_range%7])
		tt.down()
		tt.begin_fill()
		tt.circle(cm_radius - i, cm_angle)
		tt.lt(cm_angle)
		tt.circle(cm_radius - i, cm_angle)
		tt.rt(cm_angle)
		for j in range(2):
			tt.lt(cm_radius)
			tt.rt(20)
		tt.end_fill()
		tt.up()
#--------------------------------------
def flower_center(rng, x, y):
#--------------------------------------
	tt.up()
	tt.setpos(x, y)
	tt.pensize(2)
	h=0.5
	tt.bgcolor("black")

	for i in range(rng):
		c = hsv_to_rgb(h, 1, 1)
		tt.color(c)
		h+=.004
		tt.down()
		tt.begin_fill()
		tt.circle(200 - i, 100)
		tt.lt(100)
		tt.circle(200 - i, 100)
		tt.rt(100)
		for j in range(4):
			tt.lt(200)
			tt.rt(20)
		tt.end_fill()
		tt.up()
#--------------------------------------
def flower_buzzsaw_size(radius):
#--------------------------------------
	tt.pencolor("#5E2612")
	tt.circle(-radius, 45); tt.lt(135);
	tt.circle(radius*0.25, 60); 
	tt.circle(-radius*0.25,-60); 

#--------------------------------------
def flower_buzzsaw(radius, amt_petals, x, y):
#--------------------------------------
	tt.speed(1000)
	#----------------------------------
	# Suppress drawing STEM
	#----------------------------------
	tt.up();
	tt.goto(x, y)
	tt.down()
	tt.pensize(4)
	tt.lt(90)
	radius = radius*1.2
	tt.lt(90)
	tt.color(colors_flowers_blue, "red") #Askew20220302 "limegreen")
	tt.begin_fill()
	for i in range(amt_petals):
		flower_buzzsaw_size(radius)
		tt.lt(360/amt_petals)
	tt.end_fill()
	tt.up()
	tt.ht()

#--------------------------------------
def star5_flower(rng, factor, x, y, fg, bg):
#--------------------------------------
    star.up()
    tt.home()
    star.goto(x, y)
    star.down()
    for i in range(star5_rng):
    	star.color(fg, bg)
    	star.begin_fill()
    	star.circle(factor-i, 90)
    	star.end_fill()
    	star.left(90)
    	star.begin_fill()
    	star.circle(factor-i, 90)
    	star.end_fill()
    	star.left(18)
    star.up()
#--------------------------------------
def star5_flower2(rng, factor, x, y, fg, bg):
#--------------------------------------
    star.up()
    tt.home()
    star.goto(x, y)
    star.down()
    for i in range(star5_rng):
    	star.color(fg, bg)
    	star.begin_fill()
    	star.circle(factor-i, 90)
    	star.end_fill()
    	star.left(90)
    	star.color(bg, fg)
    	star.begin_fill()
    	star.circle(factor-i, 90)
    	star.end_fill()
    	star.left(18)
    star.up()


#--------------------------------------
def carnation_multicolor(rng, angle, x, y):
#--------------------------------------
    tt.up()
    tt.home()
    star.up()
    star.goto(x, y)
    for i in range(rng):
    	tt.pencolor = carnation_multicolor_colors[rng%len(carnation_multicolor_colors)]
    	star.fillcolor(tt.pencolor)
    	star.down()
    	star.forward(i)
    	star.left(angle)
    	star.begin_fill()
    	star.circle(20)
    	star.end_fill()
    star.up()
#--------------------------------------
def drawCenter(x, y):
#--------------------------------------
    tt.up()
    tt.goto(-300, -250) 
    tt.setheading(0) 
    #tt.goto(x, y)
    center_radius = 20
    star.pensize(4)
    tt.down()
    for loop in range((4*(len(center_color)))):
    	star.color(center_color[loop%(len(center_color))])
    	for i in range(6):
    		star.circle(center_radius)
    		
    		star.right(60)
    	center_radius = center_radius + 2
    star.pensize(1) # Restore the pensize back to default
#--------------------------------------
def drawCarnation(d, angle, x, y):
#--------------------------------------
	c = 0
	tt.setposition(0, -300)
	tt.setheading(90)
	star.up()
	star.goto(x, y)
	star.down()
	for i in range(1, 1000):
		star.pencolor(colors_carnation[c])
		star.forward(d)
		star.left(angle)
		d = d - .005
		c =i%(len(colors_carnation))


#--------------------------------------
def sunflower(iter,diskRat, factor, x, y):
#--------------------------------------
    tt.home()
    sunf_iter = iter
    sunf_diskRatio = diskRat
    sunf_factor = factor + math.sqrt(1.25)
    sunf_x = x
    sunf_y = y
    sunf_max_Radius = pow(sunf_iter,sunf_factor)/sunf_iter;
    for i in range(sunf_iter+1):
	        sunf_radius = pow(i,sunf_factor)/sunf_iter;

	        if sunf_radius/sunf_max_Radius < .2:
	            star.pencolor("black")
	        if sunf_radius/sunf_max_Radius < .3:
	            star.pencolor((118,255,0)) # green
	        elif sunf_radius/sunf_max_Radius < .34:
	            star.pencolor('#5E2612') #sepia
	        elif sunf_radius/sunf_max_Radius < .4: #.4:
	            star.pencolor("red") #
	        elif sunf_radius/sunf_max_Radius < sunf_diskRatio:
	            star.pencolor('#5E2612') #sepia
	        elif sunf_radius/sunf_max_Radius == sunf_diskRatio:
	            star.pencolor('#FFA500') #orange
	        elif sunf_radius/sunf_max_Radius < .85:
	            star.pencolor('#FFA500') #orange
	        elif sunf_radius/sunf_max_Radius < .95:
	            star.pencolor('#FFC125') #goldenrod
	        else:
	            star.pencolor('#FFD700') #gold1
	     
	        theta = 2*math.pi*sunf_factor*i;
	        star.up()
	        star.setposition(sunf_x + sunf_radius * math.sin(theta), sunf_y + sunf_radius * math.cos(theta))
	        star.down()
	        star.circle(10.0 * i/(1.0*sunf_iter)) 


#--------------------------------------
def drawCircle(rng, angle, len, fg, bg, x, y):
#--------------------------------------
    star.up()
    tt.setposition(0, -300)
    tt.setheading(90)
    star.goto(x, y)
    star.down()
    star.color(bg, fg)
    star.begin_fill()
    for i in range(rng):
    	star.forward(len)
    	star.left(math.sin(i/10) * 25)
    	star.left(angle) #(i%90)
    star.end_fill()

#--------------------------------------
def star_center(rng, angle, fg, bg, x, y):
#--------------------------------------
    star.home()
    star.goto(x, y)
    star.color(fg, bg)
    star.begin_fill()
    star.down()
    for i in range(rng):
    	star.forward(math.sqrt(i) * 10)
    	star.left(angle) #(i%90)
    star.end_fill()
    star.up()
#--------------------------------------
def swirl(size,angle, x, y):
#--------------------------------------
    tt.goto(x, y)
    tt.down()
    star.begin_fill()
    for i in range(size):
	    tt.color(colors_swirl[i%5])
	    tt.pensize(i/10 + 1)
	    tt.forward(i)
	    tt.left(angle)
    star.end_fill()
    tt.up()

#--------------------------------------
def swirl2(size,angle, x, y):
#--------------------------------------
    tt.up()
    tt.goto(x, y)
    tt.down()
    star.begin_fill()
    for i in range(size):
	    tt.color(colors_swirl2[i%5])
	    tt.pensize(i/10 + 1)
	    tt.forward(i)
	    tt.left(angle)
    star.end_fill()
    tt.up()


#--------------------------------------
def swirl3(size,angle, x, y):
#--------------------------------------
    tt.up()
    tt.goto(x, y)
    tt.down()
    star.begin_fill()
    for i in range(size):
	    tt.color(colors_swirl3[i%5])
	    tt.pensize(i/10 + 1)
	    tt.forward(i)
	    tt.left(angle)
    star.end_fill()
    tt.up()

#--------------------------------------
def swirl4(size,angle, x, y):
#--------------------------------------
    tt.up()
    tt.goto(x, y)
    tt.down()
    star.begin_fill()
    for i in range(size):
	    tt.color(colors_swirl4[i%5])
	    tt.pensize(i/10 + 1)
	    tt.forward(i)
	    tt.left(angle)
    star.end_fill()
    tt.up()

#--------------------------------------
def drawStar(d, angle, x, y):
#--------------------------------------
	c = 0
	star.up()
	star.goto(x, y)
	star.down()
	#for i in range(1, 400):
	for i in range(1, 1200):
		star.pencolor(colors_star[c])
		star.forward(d)
		star.left(angle)
		d = d - 2 
		c =i%4
	star.up()

#--------------------------------------
def drawRose(d, angle, x, y):
#--------------------------------------
    star.up()
    tt.setposition(0, -300)
    tt.setheading(90)
    star.goto(x, y)
    star.down()
    star.begin_fill()
    for i in range(1, 53):
    	star.forward(d)
    	star.left(angle)
    	d = d -5
    star.end_fill()
    star.up()

#--------------------------------------
def drawSmallRose(d, angle, x, y):
#--------------------------------------
    star.up()
    tt.setposition(0, -300)
    tt.setheading(90)
    star.goto(x, y)
    star.down()
    star.begin_fill()
    for i in range(1, 80):
    	star.forward(d)
    	star.left(angle)
    	d = d -5
    star.end_fill()
    star.up()
#--------------------------------------
def title():
#--------------------------------------
	tt.up()
	tt.goto(-420,385)
	tt.color('#EC008C')
	tt.write('Flowers for Elizabeth', align='center', font=('TIMES', 10, 'bold','underline'))
	tt.down()
#======================================
# Write Title first
#======================================
title()

#--------------------------------------
# TINY flower in upper right
#--------------------------------------
arcFlower(250, 2, 170, 50)
#======================================
# Create Sunflower
#======================================
sunflower(3000, .7, .5,   0,  200) # sunf top center # was y=225
sunflower(3000, .7, .5, 200, -200) # sunf bottom right 
sunflower(3000, .7, .5,-200, -200) # sunf bottom left
#--------------------------------------
# Add 5-Star Green Blue
#--------------------------------------
star5_flower(30, 75,  50, -50, (0, 0, 255), (118, 238, 0)) #Askew20220304
#--------------------------------------
# Swirl2
#--------------------------------------
swirl2(100,59, -55, 100)  #Askew20220228
#=====================================
# Swirl Teal Green
#=====================================
# Add teal/mint swirl in upper left corner
#--------------------------------------
swirl4(100,59, -145, 10) # Askew20220228
#--------------------------------------
# Blue Swirl3
#--------------------------------------
swirl3(100,59, -275, 50) #Askew20200228
#--------------------------------------
# Draw Small Rose
#--------------------------------------
star.color("black", colors_rose[2])
drawSmallRose(250, 250,  75, 250)
drawSmallRose(250, 250,-225, 300)
drawSmallRose(250, 250, 100,-140) # Rose at bottom of page

#--------------------------------------
# Buzzsaw Flower on the top and right
#--------------------------------------
tt.setheading(0)
flower_buzzsaw(100, 7, 0, 100)
tt.setheading(0)
flower_buzzsaw(100, 7, 170, -50) 
star.goto(0.00, 0.00)
star.setheading(348.4407626773)
# Compensate for suppressed drawCirle
star.setheading(126.8815253549)   # Compensate
star.setposition(222.96,81.88) # Compensate  
#======================================
# Start Roses
#======================================
star.color("black", colors_rose[0])
drawRose(100, 98, 300, 10) 
#--------------------------------------
# Put a center in the violet flower
#--------------------------------------
star_center(35, 170, colors_flowers_yellow, '#822222', 240, 20)
#--------------------------------------
# Rose Violet
#--------------------------------------
star.color("black", colors_rose[0])
drawRose(100, 98,-150, 50)
#--------------------------------------
# Put a center in the violet flower
#--------------------------------------
star_center(35, 170, colors_flowers_yellow, '#822222',-190, 0)
#--------------------------------------
# Small Rose Yellow
#--------------------------------------
star.color("black", '#FFD700') 
drawRose(55, 98, -225, -75) 
#--------------------------------------
# Center in the Small Rose Yellow
#--------------------------------------
star_center(55, 170, "brown", "black", -245,  -100) 
# #======================================
# # Put Sun at top left
# #======================================
star_center(100, 170, colors_flowers_yellow, "pink", -330, 350) 
#--------------------------------------
# Swirl
#--------------------------------------
swirl(100,59, 75, 105)
#--------------------------------------
#--------------------------------------
swirl4(100,59, 150, -100) 
#--------------------------------------
# Buzzsaw Flower on the left
#--------------------------------------
flower_buzzsaw(100, 7, -50, -120)
#--------------------------------------
# Regular and Blue Swirl Bottom Center
#--------------------------------------
swirl(100,59, -15, -175) 
swirl3(100,59, -150, -150) 
star5_flower2(30, 75, 300, 0, '#FFD700', colors_flowers_blue) 

#======================================
tt.done()
#======================================

