import os, sys

import subprocess


try:

    import pytesseract

except:

    subprocess.call([('python-m pip install pytesseract')])
    #os.system('pip install pytesseract')

    import pytesseract

try:

    import PIL

except: 

    os.system('pip instal PIL')

    import PIL

try:

    from PIL import Image

except ImportError:
    
    from PIL import Image

file_name = fr'C:\\Users\\User\\Desktop\\python\\Classes\\Day3\\Code\Tools\\job_links_in_a_pic.jpeg' 
#C:\\Users\\User\\Downloads\\Adena#2_tesseract_session#1.png'

pytesseract.pytesseract.tesseract_cmd = r'C:\\app\\Tesseract\\tesseract.exe' # testing/eurotext-eng -l eng pdf'

if os.path.exists(file_name):
    mu_image=Image.open(file_name)

    print(pytesseract.image_to_string(mu_image))
