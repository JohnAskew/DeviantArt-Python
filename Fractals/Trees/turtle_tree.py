import os, sys
try:
	from random import randint
except:
	os.syste('pip install random')
	from random import randint
try:
	import turtle as tt
except:
	os.system('pip install turtle')
	import turtle as tt
thick = 16
axiom = "22220"
axmTemp = ""
itr = 12 
angl = 16 
dl = 10 
stc = []
randint_low = 0
randint_high = 10
#--------------------------------------
# Housekeeping
#--------------------------------------
translate = {"1": "21",
             "0": "1[-20]+20"}
#--------------------------------------
# Start Turtle
#--------------------------------------
tt.hideturtle()
tt.tracer(0)
tt.pencolor('#490000')
tt.penup()
tt.setposition(0, -200)
tt.pensize(thick)
tt.pendown()
tt.setheading(-270)
tt.bk(200)
tt.forward(200)
tt.setposition(0, -200)

for k in range(itr):
	for ch in axiom:
		if ch in translate:
		    axmTemp += translate[ch]
		else:
			axmTemp += ch

	axiom = axmTemp
	axmTemp = ""

for ch in axiom:
	if   ch == "+":
		tt.right(angl - randint(-13, 13)) #90))
	elif ch == "-":
		tt.left(angl - randint(-13, 13 )) #90))
	elif ch == "2": # Branches
		if randint(randint_low,randint_high) >  4: # 5: #3: #5 #4:
		    tt.forward(dl)
	elif ch == "1":
		if randint(randint_low, randint_high) > 6: #4: $5: # 4: #3: #5 #4:
			tt.forward(dl)
	elif ch == "0":
		stc.append(tt.pensize())
		tt.pensize(4)
		r = randint(randint_low, randint_high)

		if r  == 0:
			tt.pencolor('#009900') #true green
		elif r  == 1:
			tt.pencolor('#4F9900') #pine green
		elif r == 2:
			tt.pencolor('#ED9121') #carrot orange
		elif r == 3:
			tt.pencolor('#D2691E') # Chocolate brown('#490000')
		elif r == 4:
			tt.pencolor('#C0A000') #'#551D00')
		elif r == 5:
			tt.pencolor('#CD2626') #firebrick red
		elif r == 6:
			tt.pencolor('#667900') # olive(drab) green
		elif r == 7:
			tt.pencolor('#EEAD0E') #Goldenrod dark yellow
		elif r == 8:
			tt.pencolor('#BDB76B') #Khaki green
		elif r == 9:
			tt.pencolor('#DC143C') #Crimson
		else:
			tt.pencolor('#20BB00') #Lime green

		tt.forward(dl) 
		tt.pensize(stc.pop())
		tt.pencolor('#490000')
	elif ch == "[":
		thick = thick * 0.75
		tt.pensize(thick)
		stc.append(thick)
		stc.append(tt.xcor())
		stc.append(tt.ycor())
		stc.append(tt.heading())
	elif ch == "]":
		tt.penup()
		tt.setheading(stc.pop())
		tt.sety(stc.pop())
		tt.setx(stc.pop())
		thick = stc.pop()
		tt.pensize(thick)
		tt.pendown()

tt.update()
tt.mainloop()





