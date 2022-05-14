

Width  = 1000
Height = 1000

max_x = 1.6
max_y = 1.0
offset_x = -1.6
offset_y = -1.0

borders = 1.0
iterations = 500
color_mode = 2 #1
complex_number = complex(-0.8, 0.156)

Height = (max_y - offset_y)*float(Width)/(max_x - offset_x)
Height = int(Height)

size = Width, Height