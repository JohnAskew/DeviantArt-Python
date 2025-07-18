Friday 16 June 2023

Max's Python Scripts. 
I have a folder where I keep my python scripts in their own subfolders
I open VS Code in the main Python folder and then work in the subfolders. 
Thismeans tthat my current folder is not where the project code resides. 
So I need to put a PATH to the subfolder. 
So usually, at the top of the scrippt file you will usiually find a couple of lines as follows so that I can access my code: 

path = 'History/BDM/'
fileIn = '_123.csv'
fileOut = '_ABC.csv'


later I may have code like this: 
df=pd.read_csv(path + fileIn, header=0)

It is easily to fix if you are running the code from a specific directory, just change path =''   empty string:

path =  ''     #'History/BDM/'
fileIn = '_123.csv'
fileOut = '_ABC.csv'


I usualluy put an '_' or '0' at the beginning of the input/output files so they are at the top of the files in the folder, so easy to find without scolling down when looking for output or input data.


This is what I do when I load my script to the server and run it from there, 
kind regards, 
Max Drake