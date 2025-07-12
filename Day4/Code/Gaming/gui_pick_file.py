import os, sys

# Import Library
from tkinter import *
import os
from tkinter.filedialog import askopenfilename 

# Create Object 
root = Tk() 

# Set geometry
root.geometry('200x200') 

def open_file(): 
    file = askopenfilename()
    os.system('"%s"' % file)

Button(root, text ='Open', 
       command = open_file).pack(side = TOP, 
                                 pady = 10) 

# Execute Tkinter
root.mainloop() 