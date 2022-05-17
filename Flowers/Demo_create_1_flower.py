import os, sys
try:
	import turtle as tt
except:
	os.system('pip install turtle')
	import turtle as tt

t = tt.Turtle()

list1 = ["red", "yellow", "pink", "red", "pink"] 
for i in range(100):
	t.color(list1[i%5])
	t.pensize(i/10 + 1)
	t.forward(i)
	t.left(59)

tt.done()