import os, sys

# Import Module
from tkinter import *

try:
    import subprocess
except:
    os.system('pip install subprocess')
    import subprocess

# create root window
root = Tk()

# root window title and dimension
root.title("Welcome to my Pong GUI")
# Set geometry(widthxheight)
root.geometry('350x200')

# adding menu bar in root window
# new item in menu bar labelled as 'New'
# adding more items in the menu bar 
menu = Menu(root)
item = Menu(menu)
item.add_command(label='New')
menu.add_cascade(label='File', menu=item)
root.config(menu=menu)

# adding a label to the root window
lbl = Label(root, text = "Are you and the other player ready?")
lbl.grid()

# adding Entry Field
txt = Entry(root, width=10)
txt.grid(column =1, row =0)


def backend_function_with_subprocess():
        # ... some backend logic ...
        print("Backend logic complete. Launching GUI as a separate process.")

        # Path to your Tkinter application script
        tkinter_app_path = "./pong_turtle.py"

        # Launch the Tkinter app
        subprocess.Popen(["python", tkinter_app_path])

# function to display user text when
# button is clicked
def clicked():
    if txt.get() == "yes":
        backend_function_with_subprocess()
    res = "You answered: " + txt.get()
    lbl.configure(text = res)
    print(f'{txt.get()=}')
    

# button widget with red color text inside
btn = Button(root, text = "Click me" ,
             fg = "red", command=clicked)
# Set Button Grid
btn.grid(column=2, row=0)

# Execute Tkinter
root.mainloop()