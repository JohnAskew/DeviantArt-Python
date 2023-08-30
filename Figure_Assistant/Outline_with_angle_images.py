import os,sys

try:
    import triangle as tr

except:

    os.system('pip install triangle')

    import triangle as tr

try:
    
    import random

except:

    os.system('pip install random')

    import random

try:

    import PIL
    from PIL import Image, ImageFilter, ImageEnhance, ImageDraw

except:

     os.system('pip install --upgrade Pillow')

     import PIL

     from PIL import Image, ImageFilter, ImageEnhance, ImageDraw


try:
    from mpl_toolkits.mplot3d import Axes3D

except:
    os.system('pip install mpl_toolkits')

    from mpl_toolkits.mplot3d import Axes3D

from PIL import ImageFont

try:

    import cv2

except:

    os.system('pip install opencv-contrib-python')

    import cv2

try:

    from cvzone.PoseModule import PoseDetector

except:

    os.system('pip install cvzone==1.5,6')

    from cvzone.PoseModule import PoseDetector

try:

    import mediapipe as mp

except:

    os.system('pip install mediapipe')

    import mediapipe as mp

try:

    import math as m

except:

    os.system('pip install math')

    import math as m

try:
    import cmath

except:

    os.system('pip install cmath')

    import cmath

try:

    import matplotlib.pyplot as plt

except:

    os.system('pip import matplotlib')

    import matplotlib.pyplot as plt

try:

    from mediapipe import solutions

except:

    os.system('pip install mediapipe')

    from mediapipe import solutions

try:

    import numpy as np

except:

    os.system('pip install numpy')

    import numpy as np

###############################
###############################
# FUNCTIONS
###############################
###############################
#--------------------------------------
def bring_in_image():
#--------------------------------------
    return 'image3.jpg'
    #return 'free_image3.jpg'

###############################
# FUNCTIONS: Ways to calculate Rotation
###############################
#--------------------------------------
def get_torso_rotation_offset_rect(x, y, rotate_length, torso_rotation):
#--------------------------------------
    return point_pos_rect([x, y], rotate_length, torso_rotation)
#######################################
# NEXT 2 Functions are used as 1
#######################################
#--------------------------------------
def rotate_points_1_rotate(pt, radians, origin):
#--------------------------------------
    print("rotate_points_1_rotate RECEIVING ", pt, radians, origin)
    x, y = pt
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
    print("rotate_points_1_rotate RETURNING ", qx, qy)
    return qx, qy

#--------------------------------------
def rotate_points_2_offsets(x1, y1, x2, y2, width, height):
#--------------------------------------
    origin = (width/2, height/2)
    x1_new, y1_new = rotate([x1, y1], np.radians(angle), origin)
    x2_new, y2_new = rotate([x2, y2], np.radians(angle), origin)
    h_new, w_new = image.shape[:2]
    xoffset, yoffset = (w_new - width)/2, (h_new - height)/2
    x1_new, y1_new = x1_new+xoffset, y1_new+yoffset
    x2_new, y2_new = x2_new+xoffset, y2_new+yoffset
    return [x1_new, y1_new], [x2_new, y2_new]
###############################
# FUNCTIONS: Ways to calculate Angle
###############################
#---------------------------------------
def absAngle(a):
#--------------------------------------
#this yields correct counter-clock-wise numbers, like 350deg for -370
  return ((360 + (a % 360)) % 360)
#--------------------------------------
def angleDelta(a, b):
#--------------------------------------
   # Negative angles = clockwise rotation, else counterclockwise
   sign = 0
   # delta = np.abs(absAngle(a) - absAngle(b))
   delta = (absAngle(a) - absAngle(b))

   if (absAngle(a) > absAngle(b)): 
       sign = -1
       print("angleDelta delta", delta, "rotation: Clockwise")
   #(delta >= 180):
   else:
       sign = 1
       print("angleDelta delta", delta , "rotation: CounterClockwise")

   return ((180 - np.abs(delta - 180)) * sign)
#--------------------------------------
def findAngle(x1, y1, x2, y2):
#--------------------------------------
# Calculate angle.
    theta = m.acos( (y2 -y1)*(-y1) / (m.sqrt(
        (x2 - x1)**2 + (y2 - y1)**2 ) * y1) )
    degree = int(180/m.pi)*theta
    return degree
#--------------------------------------
def angle_between(p1, p2):
#--------------------------------------
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    angle_between = np.rad2deg((ang1 - ang2) % (2 * np.pi))
    return int(round(angle_between))
#--------------------------------------
def calculate_angle(a,b,c):
#--------------------------------------
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return int(angle)

#--------------------------------------
def calculate_angle_numpy(a, b, c):
#--------------------------------------
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


###############################
# FUNCTIONS: Ways to calculate Distance or Line Segment Length
###############################
#--------------------------------------
def findDistance(x1, y1, x2, y2):
#--------------------------------------
    dist = m.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist
#--------------------------------------
def point_pos(x0, y0, d, theta):
#--------------------------------------
    # theta_rad = np.pi/2 - m.radians(theta)
    theta_rad = np.pi/4 - m.radians(theta)
    return x0 + d*m.cos(theta_rad), y0 + d*m.sin(theta_rad)
#--------------------------------------
def point_pos_rect(p, d, theta):
#--------------------------------------
    x = p[0]
    y = p[1]
    # Convert to complex number (imaginary)
    complex_offset = cmath.rect(d, np.pi/2-m.radians(theta))
    # Return converted complex point into decimal
    return (x + cmath.polar(complex_offset)[0], y+ cmath.polar(complex_offset)[1])
#--------------------------------------
def get_coords(x, y, angle, imwidth, imheight):
#--------------------------------------
    x1_length = (x-imwidth) / m.cos(angle)
    y1_length = (y-imheight) / m.sin(angle)
    length = max(abs(x1_length), abs(y1_length))
    endx1 = x + length * m.cos(m.radians(angle))
    endy1 = y + length * m.sin(m.radians(angle))

    x2_length = (x-imwidth) / m.cos(angle+180)
    y2_length = (y-imheight) / m.sin(angle+180)
    length = max(abs(x2_length), abs(y2_length))
    endx2 = x + length * m.cos(m.radians(angle+180))
    endy2 = y + length * m.sin(m.radians(angle+180))

    return endx1, endy1, endx2, endy2
#--------------------------------------
def left_extract_points(start_point_x, start_point_y, angle, length):
#--------------------------------------
    P1 = (0,0)
    P1 = (start_point_x, start_point_y)
    end_point_x  =  (int(round(P1[0]  + length *  -np.cos(angle * np.pi / 180.0)))) #-np.cos(l_solar_plexus_angle * np.pi / 180.0))))
    end_point_y  =  (int(round(P1[1]  + length *  np.sin(angle * np.pi / 180.0))))
    return (end_point_x, end_point_y)
#--------------------------------------
def right_extract_points(start_point_x, start_point_y, angle, length):
#--------------------------------------
    P1 = (0,0)
    P1 = (start_point_x, start_point_y)
    end_point_x  =  (int(round(P1[0]  + length *  np.cos(angle * np.pi / 180.0)))) #-np.cos(l_solar_plexus_angle * np.pi / 180.0))))
    end_point_y  =  (int(round(P1[1]  + length *  np.sin(angle * np.pi / 180.0))))
    return (end_point_x, end_point_y) 
#--------------------------------------
def midpoint(p1, p2):
#--------------------------------------
    return int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)

#--------------------------------------
def rect(r, theta):
#--------------------------------------
    """theta in degrees

    returns tuple; (float, float); (x,y)
    """
    x = r * math.cos(math.radians(theta))
    y = r * math.sin(math.radians(theta))
    return x,y
#--------------------------------------
def polar(x, y):
#--------------------------------------
    """returns r, theta(degrees)
    """
    r = (x ** 2 + y ** 2) ** .5
    theta = math.degrees(math.atan2(y,x))
    return r, theta


#--------------------------------------
def polar2cart(r, theta):
#--------------------------------------
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y
#--------------------------------------
def cart2polar(x, y):
#--------------------------------------
    rho  = np.sqrt(x**2 + y**2) 
    phi = np.arctan(y/x)
    return rho, phi
#--------------------------------------
def make_box(x, y, w, h, attribute):
#--------------------------------------

    i = len(vertices)
    
    vertices.extend([[x,   y],
                     [x+w, y],
                     [x+w, y+h],
                     [x,   y+h]])

    segments.extend([(i+0, i+1),
                     (i+1, i+2),
                     (i+2, i+3),
                     (i+3, i+0)])
    
    regions.append([x+0.5*w, y+0.5*h, attribute, 0])
#--------------------------------------
# Check if a point is inside a rectangle
def rect_contains(rect, point) :
#--------------------------------------
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

#--------------------------------------
# Draw a point
def draw_point(img, p, color ) :
#--------------------------------------
    cv2.circle( img, p, 2, color, cv2.FILLED, cv2.LINE_4, 0 )

#--------------------------------------
# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color ) :
#--------------------------------------
 
    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])
 
    for t in triangleList :
 
        pt1 = int(t[0]), int(t[1]) #(t[0], t[1])
        pt2 = int(t[2]), int(t[3])
        pt3 = int(t[4]), int(t[5])
        print("Delauney code pt1", pt1, "pt2", pt2, "pt3", pt3, "r", r)

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
 
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_4, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_4, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_4, 0)

#--------------------------------------
# Visualize Right Side
def visualize_right_side(image
#--------------------------------------
    , head_center_x
    , head_center_y
    , c_shldr_x
    , c_shldr_y
    , c_breast_x
    , c_breast_y
    , center_torso
    , c_navel_x
    , c_navel_y
    , c_hip_x
    , c_hip_y
    , r_knee_x 
    , r_knee_y
    , torso_width
    , torso_height
    , width
    , height
    , torso_rotation_offset_rect
    , center_line_offset_x
    , center_line_offset_y
    , center_line_green
    , center_line_color):
    print("RIGHT_HIP higher --> Right side bears weight")
    cv2.line(image, (c_shldr_x, c_shldr_y)
        # torso_rotation_offset_rect_points
    , ((c_breast_x + int(round(torso_rotation_offset_rect[0]/width)*torso_width))
    ,  (c_breast_y+ int(round(torso_rotation_offset_rect[1]/height)*torso_height)))
    ,  center_line_green, 5)

    cv2.line(image, ((c_breast_x+int(round(torso_rotation_offset_rect[0]/width)*torso_width))
    , (c_breast_y+ int(round(torso_rotation_offset_rect[1]/height)*torso_height)))
    , ((int(torso_rotation_offset_rect[0])
    , int(torso_rotation_offset_rect[1])))
    , center_line_green, 5)

    cv2.line(image, (((
    center_torso[0] +int(round(torso_rotation_offset_rect[0]/width)*torso_width))
    , (center_torso[1] + int(round(torso_rotation_offset_rect[1]/height)*torso_height))))
    , ((c_navel_x +int(round(torso_rotation_offset_rect[0]/width)*torso_width))
    , ((c_navel_y+ int(round(torso_rotation_offset_rect[1]/height)*torso_height))))
    , center_line_green, 5)

    cv2.line(image, ((c_navel_x + int(round(torso_rotation_offset_rect[0]/width)*torso_width))
    , (c_navel_y + int(round(torso_rotation_offset_rect[1]/height)*torso_height)))
    , (c_hip_x, c_hip_y), center_line_green, 5) #Askew20230822
    
    cv2.line(image, (c_hip_x , c_hip_y), (r_hip_x, r_hip_y),  center_line_green, 5) #Askew20230822
    cv2.line(image, (r_hip_x , r_hip_y), (r_knee_x, r_knee_y),  center_line_green, 5) #Askew20230822
    cv2.line(image, (r_knee_x , r_knee_y), (r_ankle_x, r_ankle_y),  center_line_green, 5) #dark_blue, 2)
    if r_foot_index2center > l_foot_index2center: #c_heel_x:
        cv2.line(image, ((head_center_x -center_line_offset_x), (head_center_y - center_line_offset_y)), (l_foot_x, l_foot_y), center_line_color, 1)
    elif l_foot_index2center > r_foot_index2center: #c_heel_x:
        cv2.line(image, ((head_center_x -center_line_offset_x), (head_center_y - center_line_offset_y)), (r_foot_x, r_foot_y), center_line_color, 1)
    else:
        cv2.line(image, ((head_center_x -center_line_offset_x), (head_center_y - center_line_offset_y)), (c_knee_x, c_knee_y), center_line_color, 1)
#------------------------------
def visualize_left_side(image
#------------------------------
    , head_center_x
    , head_center_y
    , c_shldr_x
    , c_shldr_y
    , c_breast_x
    , c_breast_y
    , center_torso
    , c_navel_x
    , c_navel_y
    , c_hip_x
    , c_hip_y
    , l_knee_x 
    , l_knee_y
    , torso_width
    , torso_height
    , width
    , height
    , torso_rotation_offset_rect
    , center_line_offset_x
    , center_line_offset_y
    , center_line_green
    , center_line_color):
#------------------------------
    print("LEFT_HIP higher --> Left Leg bears weight")
    cv2.line(image, (c_shldr_x, c_shldr_y)
    , ((c_breast_x - int(round(torso_rotation_offset_rect[0]/width)*torso_width))
    ,  (c_breast_y- int(round(torso_rotation_offset_rect[1]/height)*torso_height)))
    ,  center_line_green, 5)

    cv2.line(image, ((c_breast_x-int(round(torso_rotation_offset_rect[0]/width)*torso_width))
    , (c_breast_y- int(round(torso_rotation_offset_rect[1]/height)*torso_height)))
    , ((center_torso[0] -int(round(torso_rotation_offset_rect[0]/width)*torso_width))
    , (center_torso[1]- int(round(torso_rotation_offset_rect[1]/height)*torso_height)))
    , center_line_green, 5)

    cv2.line(image, (((
    center_torso[0] -int(round(torso_rotation_offset_rect[0]/width)*torso_width))
    , (center_torso[1] - int(round(torso_rotation_offset_rect[1]/height)*torso_height))))
    , ((c_navel_x -int(round(torso_rotation_offset_rect[0]/width)*torso_width))
    , ((c_navel_y- int(round(torso_rotation_offset_rect[1]/height)*torso_height))))
    , center_line_green, 5)

    cv2.line(image, ((c_navel_x - int(round(torso_rotation_offset_rect[0]/width)*torso_width))
    , (c_navel_y - int(round(torso_rotation_offset_rect[1]/height)*torso_height)))
    , (c_hip_x, c_hip_y), center_line_green, 5) #Askew20230822
    cv2.line(image, (c_hip_x , c_hip_y), (l_hip_x, l_hip_y),  center_line_green, 5) #Askew20230822
    cv2.line(image, (l_hip_x , l_hip_y), (l_knee_x, l_knee_y),  center_line_green, 5)
    cv2.line(image, (l_knee_x , l_knee_y), (l_ankle_x, l_ankle_y),  center_line_green, 5)
    if r_foot_index2center > l_foot_index2center: #c_heel_x:
        cv2.line(image, ((head_center_x -center_line_offset_x), (head_center_y - center_line_offset_y)), (l_foot_x, l_foot_y), center_line_color, 1)
    elif l_foot_index2center > r_foot_index2center: #c_heel_x:
         cv2.line(image, ((head_center_x -center_line_offset_x), (head_center_y - center_line_offset_y)), (r_foot_x, r_foot_y), center_line_color, 1)
    else:
         cv2.line(image, ((head_center_x -center_line_offset_x), (head_center_y - center_line_offset_y)), (c_knee_x, c_knee_y), center_line_color, 1)
    #cv2.line(image, (head_center_x - center_line_offset_x, head_center_y - center_line_offset_y), (l_heel_x, l_heel_y),  center_line_color, 1)
###############################
# Classes
###############################
class Point(object):
#------------------------------
    def __init__(self, x=None, y=None, r=None, theta=None):
        """x and y or r and theta(degrees)
        """
        if x and y:
            self.c_polar(x, y)
        elif r and theta:
            self.c_rect(r, theta)
        else:
            raise ValueError('Must specify x and y or r and theta')
    def c_polar(self, x, y, f = polar):
        self._x = x
        self._y = y
        self._r, self._theta = f(self._x, self._y)
        self._theta_radians = math.radians(self._theta)
    def c_rect(self, r, theta, f = rect):
        """theta in degrees
        """
        self._r = r
        self._theta = theta
        self._theta_radians = math.radians(theta)
        self._x, self._y = f(self._r, self._theta)
    def setx(self, x):
        self.c_polar(x, self._y)
    def getx(self):
        return self._x
    x = property(fget = getx, fset = setx)
    def sety(self, y):
        self.c_polar(self._x, y)
    def gety(self):
        return self._y
    y = property(fget = gety, fset = sety)
    def setxy(self, x, y):
        self.c_polar(x, y)
    def getxy(self):
        return self._x, self._y
    xy = property(fget = getxy, fset = setxy)
    def setr(self, r):
        self.c_rect(r, self._theta)
    def getr(self):
        return self._r
    r = property(fget = getr, fset = setr)
    def settheta(self, theta):
        """theta in degrees
        """
        self.c_rect(self._r, theta)
    def gettheta(self):
        return self._theta
    theta = property(fget = gettheta, fset = settheta)
    def set_r_theta(self, r, theta):
        """theta in degrees
        """
        self.c_rect(r, theta)
    def get_r_theta(self):
        return self._r, self._theta
    r_theta = property(fget = get_r_theta, fset = set_r_theta)
    def __str__(self):
        return '({},{})'.format(self._x, self._y)
 
###############################
# BEGIN MAIN LOGIC
###############################
#-----------------------------
# Variables
#-----------------------------
# height=640
# width=480
vertices = []
segments = []
regions = []
SHOULDER_HIGHER=""
HIP_HIGHER = ""
ROTATION_NEGATIVE = "CLOCKWISE"
ROTATION_POSITIVE = "COUNTERCLOCKWISE"
ROTATION_ZERO     =  "NO ROTATION"
ROTATION_DIRECTION = ""
ROTATE_RIGHT = "RIGHT"
ROTATE_LEFT  = "LEFT"
BEND_DIRECTION = ""
BEND_RIGHT = "RIGHT"
BEND_LEFT  = "LEFT"

HIP2SHLDR_ADJUST_PERCENT = 125 # To expand woman's skinny hips to match shoulders, if need be. 125 = increase by 25% of original.
HIP_W_2_TORSO = .185             #Ratio of hip width to torso height, per head units

#--------------------------
# Colors
#--------------------------
blue = (255, 127, 0)
red = (50, 50, 255)
red_orange = (120,0,248) #(0, 60, 175)
red_navel = (40, 20, 255)
orange = (0, 125, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
#yellow = (0, 255, 255)
purple = (180, 45, 126)
cv2_pts_olive_green = (88,117,66)
cv2_lines_royal_blue = (245,66,66)
my_pink = (255, 99, 225)
pink_salmon = (180, 90, 255)
yellow=(0,255,180)
pink = (255, 0, 255)
my_white = (248, 255, 255)
center_line_green = (80, 255, 80)
contra_lines_width = 1
center_line_color = (255, 125, 255)
center_line_offset_y = int(5) #100)
center_line_offset_x = int(abs(center_line_offset_y * .01))
image_read = bring_in_image()
minLineLength = 30
maxLineGap = 5
puttext_offset= -15
#####################################
## TO DO
#####################################
# 4. Correct gesture line in torso, using angle offset from center
#    ==> ** Correct center line for martial arts.
#        ==> Line from center head to each heel.
#        ==> The heel closest to the head center line is supporting leg.

#    * Look at twist direction and compare to center line for drawing correct center line
# 5. Maybe construct a pelvis 3-D.
#    *  # Extend Navel Line, Box in Pelvis - calc lenght as findDistance of l_navel_x, l_navel_y, r_navel_x, r_navel_y
#    * This works! length = width/findDistance(l_navel_x, l_navel_y, r_navel_x, r_navel_y) 
#    ** Add check, if navel point > solar_plexus point, calc distance using solar_plexus point, not navel point.
# 6.
#####################################
## DONE
#####################################
# 1. determine_load_bearing_leg_by_looking_for_hip_side_that_compresses
#    a. Compare left_hip to right_hip and take the highest 1
#    b. Draw from the load bearing hip to the load bearing knee with extra wide line
# 2. Using the lower solar plexus, determine the navel line and make red-orange line thru navel
# 3. Connect center lines from shoulder to breast, breast to ribcage, ribcage to navel and navel to center hips

directory=os.getcwd()
for filename in os.scandir(directory):
    #DEBUG print("filename", filename)
    if filename.is_file():
        split_tup = os.path.splitext(filename)
        file_suffix = split_tup[1]
        
        if file_suffix in ('.png', '.jpg'):
            file_processed = (split_tup[0] + '_scanned' + split_tup[1])
            #SAVE LATER DEBUG print("file_processed", file_processed)


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose



####################
# Experimental code
####################
##
## Experimental code writes to image thru cv2
##   
win_delaunay = "Delaunay Triangulation"
#win_voronoi = "Voronoi Diagram"

# Turn on animation while drawing triangles
animate = True

# Define colors for drawing.
delaunay_color = (255,255,255)
points_color = (0, 0, 255)

# Read in the image.
img = cv2.imread(image_read);
#img = cv2.resize(img, (width,height))

# Keep a copy around
img_orig = img.copy();

# Rectangle to be used with Subdiv2D
size = img.shape
rect = (0, 0, size[1], size[0])


# Create an instance of Subdiv2D
subdiv = cv2.Subdiv2D(rect);

# Create an array of points.
points = [];
#--
## Code continued near bottom 
##

#=================================
# START OPENCV
#=================================


#cap = cv2.imread(image_read) #VideoCapture(1)
cap = img
with mp_pose.Pose(min_detection_confidence=0.58, min_tracking_confidence=0.85) as pose:

    # while True: #cap.isOpened():
        #ret, frame =cap.read()
    image = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    image = cv2.resize(image, (width,height))

    image.flags.writeable = False
    
    #make detections
    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_three = cv2.merge([gray,gray,gray])

    edges = cv2.Canny(gray_three,150,200,apertureSize = 3) #3)
    lines = cv2.HoughLinesP(edges,cv2.HOUGH_PROBABILISTIC, np.pi/180, 30, minLineLength,maxLineGap)
    for x in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[x]:
            cv2.line(gray_three,(x1,y1),(x2,y2),(120,0,248),2, cv2.LINE_4)
   
    cv2.imshow('gray_three', gray_three)

 


###################
# Coninue with regular code
###################
    #Extract Landmarks
    try:
        landmarks = results.pose_landmarks.landmark
    #DEBUG print("landmarks=", landmarks) # prints x, y, z as dictionary
    except Exception as error:
        print("Issue processing 'landmarks = results.pose_landmarks.landmark. error", type(error).__name__, error)

    ######################
    # Get coordinates
    #####################
    # EARS
    #---------------------
    LEFT_EAR = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y,]
    RIGHT_EAR = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y,]
    CENTER_EAR = [(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x)/2,
                  (landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y)/2]
    #---------------------
    # EYES
    #---------------------
    # EYE OUTER
    LEFT_EYE_OUTER = [landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].y,]
    RIGHT_EYE_OUTER = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].y,]
    CENTER_EYE_OUTER = [(landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].x)/2,
                  (landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].y)/2]

    # EYE INNER
    LEFT_EYE_INNER = [landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].y,]
    RIGHT_EYE_INNER = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].y,]
    CENTER_EYE_INNER = [(landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].x)/2,
                  (landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].y)/2]
    # EYE CENTER
    LEFT_EYE = [landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y,]
    RIGHT_EYE = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y,]
    CENTER_EYE = [(landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x)/2,
                  (landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y)/2]
    #---------------------
    # NOSE
    #---------------------
    NOSE = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y,]

    #---------------------
    # Mouth
    #---------------------
    LEFT_MOUTH = [landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y,]
    RIGHT_MOUTH = [landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y,]
    CENTER_MOUTH = [(landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x + landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x)/2,
                  (landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y + landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y)/2]
    #---------------------
    # Shoulders
    #---------------------
    try:
        LEFT_SHOULDER=  [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,]
        RIGHT_SHOULDER= [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        CENTER_SHOULDER = [(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x)/2,
                  (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)/2]

    except Exception as error:
        print("Issue with processing SHOULDERS error", type(error).__name__, error)
        print("check/reduce min_detection_confidence")
  
    #---------------------
    # ELBOW
    #---------------------
    try:
        LEFT_ELBOW   =  [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,]
        RIGHT_ELBOW   = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    except Exception as error:
        print("Issue with processing ELBOWS error", type(error).__name__, error)
        print("check/reduce min_detection_confidence")


    LEFT_WRIST   =  [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,]
    RIGHT_WRIST   = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,]

    # HIPS
    LEFT_HIP  =  [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,]
    RIGHT_HIP =  [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,]
    CENTER_HIP = [(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x)/2,
                 (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y)/2]
    

    #print("landmarks with index for LEFT_HIP",landmarks[mp_pose.PoseLandmark('LEFT_HIP')].value)

    print(80 * '-')
    print("CENTER_HIP", CENTER_HIP, "CENTER_HIP_on IMAGE", tuple(np.multiply(CENTER_HIP, [width, height]).astype(int))) 
    print(80 * '-')

    LEFT_KNEE = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,]
    RIGHT_KNEE = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,]
    CENTER_KNEE = [(LEFT_KNEE[0] + RIGHT_KNEE[0])/2, (LEFT_KNEE[1] + RIGHT_KNEE[1])/2]
              
    LEFT_ANKLE = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,]
    RIGHT_ANKLE = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,]
    CENTER_ANKLE = [(LEFT_ANKLE[0] + RIGHT_ANKLE[0])/2, (LEFT_ANKLE[1] + RIGHT_ANKLE[1])/2]

    LEFT_FOOT_INDEX = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y,]
    RIGHT_FOOT_INDEX = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y,]
    CENTER_FOOT_INDEX = [(LEFT_FOOT_INDEX[0] + RIGHT_FOOT_INDEX[0])/2, (LEFT_FOOT_INDEX[1] + RIGHT_FOOT_INDEX[1])/2]

    LEFT_HEEL = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y,]
    RIGHT_HEEL = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y,]
    CENTER_HEEL = [(LEFT_HEEL[0] + RIGHT_HEEL[0])/2, (LEFT_HEEL[1] + RIGHT_HEEL[1])/2] 
    
    
    # Solar Plexus
    LEFT_SOLAR_PLEXUS= [(float(LEFT_SHOULDER[0]) + float(LEFT_HIP[0])) /2
                         ,(float(LEFT_SHOULDER[1]) + float(LEFT_HIP[1])) /2]
    #[[(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x)/2], [(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)/2]]
    RIGHT_SOLAR_PLEXUS= [(float(RIGHT_SHOULDER[0]) + float(RIGHT_HIP[0])) /2
                         ,(float(RIGHT_SHOULDER[1]) + float(RIGHT_HIP[1]))/2]
    #[[(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x)/2], [(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y)/2]]
    CENTER_SOLAR_PLEXUS = ((LEFT_SOLAR_PLEXUS[0] + RIGHT_SOLAR_PLEXUS[0])/2 ,(LEFT_SOLAR_PLEXUS[1] + RIGHT_SOLAR_PLEXUS[1])/2)  

    LEFT_NAVEL =  ((LEFT_SOLAR_PLEXUS[0] + LEFT_HIP[0])/2),((LEFT_SOLAR_PLEXUS[1] + LEFT_HIP[1])/2)
    RIGHT_NAVEL =  ((RIGHT_SOLAR_PLEXUS[0] + RIGHT_HIP[0])/2),((RIGHT_SOLAR_PLEXUS[1] + RIGHT_HIP[1])/2)
    CENTER_NAVEL = ((LEFT_NAVEL[0] + RIGHT_NAVEL[0])/2, (LEFT_NAVEL[1] + RIGHT_NAVEL[1])/2)

    # Rib Cage
    LEFT_RIB_CAGE = [(float(LEFT_SOLAR_PLEXUS[0]) + float(LEFT_NAVEL[0]))/2
                    ,(float(LEFT_SOLAR_PLEXUS[1]) + float(LEFT_NAVEL[1]))/2]
    RIGHT_RIB_CAGE = [(float(RIGHT_SOLAR_PLEXUS[0]) + float(RIGHT_NAVEL[0]))/2
                    ,(float(RIGHT_SOLAR_PLEXUS[1]) + float(RIGHT_NAVEL[1]))/2]
    CENTER_RIB_CAGE = ((LEFT_RIB_CAGE[0] + RIGHT_RIB_CAGE[0])/2, (LEFT_RIB_CAGE[1] + RIGHT_RIB_CAGE[1])/2)

    # Breast Line
    LEFT_BREAST = ((LEFT_SHOULDER[0] + LEFT_SOLAR_PLEXUS[0])/2),((LEFT_SHOULDER[1] + LEFT_SOLAR_PLEXUS[1])/2)
    RIGHT_BREAST = ((RIGHT_SHOULDER[0] + RIGHT_SOLAR_PLEXUS[0])/2),((RIGHT_SHOULDER[1] + RIGHT_SOLAR_PLEXUS[1])/2)
    CENTER_BREAST = ((LEFT_BREAST[0] + RIGHT_BREAST[0])/2, (LEFT_BREAST[1] + RIGHT_BREAST[1])/2)
  
    

    #Eyes
    l_eye_outer_x, l_eye_outer_y = int(round(LEFT_EYE_OUTER[0] * width)), int(round(LEFT_EYE_OUTER[1] * height))
    r_eye_outer_x, r_eye_outer_y = int(round(RIGHT_EYE_OUTER[0] * width)), int(round(RIGHT_EYE_OUTER[1] * height))
    l_eye_inner_x, l_eye_inner_y = int(round(LEFT_EYE_INNER[0] * width)), int(round(LEFT_EYE_INNER[1] * height))
    r_eye_inner_x, r_eye_inner_y = int(round(RIGHT_EYE_INNER[0] * width)), int(round(RIGHT_EYE_INNER[1] * height))
    l_eye_x, l_eye_y = int(round(LEFT_EYE[0]   * width)), int(round(LEFT_EYE[1]   * height))
    r_eye_x, r_eye_y = int(round(RIGHT_EYE[0]  * width)), int(round(RIGHT_EYE[1]  * height))
    c_eye_x, c_eye_y = int(round(CENTER_EYE[0] * width)), int(round(CENTER_EYE[1] * height))

    # Ears
    l_ear_x, l_ear_y = int(LEFT_EAR[0] * width), int(LEFT_EAR[1] * height)
    r_ear_x, r_ear_y = int(RIGHT_EAR[0] * width), int(RIGHT_EAR[1] * height)
    c_ear_x, c_ear_y = int(CENTER_EAR[0] * width), int(CENTER_EAR[1] * height)

    # Left Mouth
    l_mouth_x, l_mouth_y = int(round(LEFT_MOUTH[0] * width)), int(round(LEFT_MOUTH[1] * height))
    r_mouth_x, right_mouth_y = int(round(RIGHT_MOUTH[0] * width)), int(round(RIGHT_MOUTH[1] * height))
    c_mouth_x, c_mouth_y = int(round(CENTER_MOUTH[0] * width)), int(round(CENTER_MOUTH[1] * height))

    # Nose
    nose_xy = int(round(NOSE[0] * width)), int(round(NOSE[1] * height))

    # Left ear.
    l_ear_x = int(landmarks[mp_pose.PoseLandmark.LEFT_EAR].x * width)
    l_ear_y = int(landmarks[mp_pose.PoseLandmark.LEFT_EAR].y * height)

     # Right ear.
    r_ear_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x * width)
    r_ear_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y * height)

    #Left shoulder
    l_shldr_x = int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width)
    l_shldr_y = int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height)
    c_shldr_x = int(CENTER_SHOULDER[0] * width)
    c_shldr_y = int(CENTER_SHOULDER[1] * height)
     
    # Right shoulder.
    r_shldr_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width)
    r_shldr_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)

    c_shldr_angle = findAngle(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
    c_shldr_angle_abs = absAngle(c_shldr_angle)


    shldr_length = width/findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y) 
     
    # Left elbow
    l_elbow_x = int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * width)
    l_elbow_y = int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * height)

    # Right elbow
    r_elbow_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * width)
    r_elbow_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * height)
     
    # Left hip.
    l_hip_x = int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * width)
    l_hip_y = int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * height)

    # Right hip.
    r_hip_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * width)
    r_hip_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * height)

    c_hip_x = int(CENTER_HIP[0] * width)
    c_hip_y = int(CENTER_HIP[1] * height)

    hip_length = findDistance(l_hip_x, l_hip_y, r_hip_x, r_hip_y)
    #-------------------------
    # Solar Plexus
    #-------------------------
    l_solar_x = int(LEFT_SOLAR_PLEXUS[0] * width)
    l_solar_y = int(LEFT_SOLAR_PLEXUS[1] * height)

    r_solar_x = int(RIGHT_SOLAR_PLEXUS[0] * width)
    r_solar_y = int(RIGHT_SOLAR_PLEXUS[1] * height)

    c_solar_x, c_solar_y = ((l_solar_x + r_solar_x)/2), (l_solar_y + r_solar_y)/2
    center_solar_x, center_solar_y = int(CENTER_SOLAR_PLEXUS[0] * width), int(CENTER_SOLAR_PLEXUS[1] * height)
    
    l_rib_cage_x = int(LEFT_RIB_CAGE[0] * width)
    l_rib_cage_y = int(LEFT_RIB_CAGE[1] * height)
    r_rib_cage_x = int(RIGHT_RIB_CAGE[0] * width)
    r_rib_cage_y = int(RIGHT_RIB_CAGE[1] * height)
    c_rib_cage_x = int(CENTER_RIB_CAGE[0] * width)
    c_rib_cage_y = int(CENTER_RIB_CAGE[1] * height)
    rib_cage_length = findDistance(l_rib_cage_x, l_rib_cage_y, r_rib_cage_x, r_rib_cage_y)/np.pi/4

    l_breast_x = int(LEFT_BREAST[0] * width)
    l_breast_y = int(LEFT_BREAST[1] * height)

    r_breast_x = int(RIGHT_BREAST[0] * width)
    r_breast_y = int(RIGHT_BREAST[1] * height)

    c_breast_x = int(CENTER_BREAST[0] * width)
    c_breast_y = int(CENTER_BREAST[1] * height)

    l_navel_x  = int(LEFT_NAVEL[0] * width)
    l_navel_y  = int(LEFT_NAVEL[1] * height)
    r_navel_x  = int(RIGHT_NAVEL[0] * width)
    r_navel_y  = int(RIGHT_NAVEL[1] * height)
    c_navel_x  = int(CENTER_NAVEL[0] * width)
    c_navel_y  = int(CENTER_NAVEL[1] * height)

    l_knee_x   = int(LEFT_KNEE[0] * width)
    l_knee_y   = int(LEFT_KNEE[1] * height)

    r_knee_x   = int(RIGHT_KNEE[0] * width)
    r_knee_y   = int(RIGHT_KNEE[1] * height)

    c_knee_x   = int(CENTER_KNEE[0] * width)
    c_knee_y   = int(CENTER_KNEE[1] * height)

    l_ankle_x  = int(LEFT_ANKLE[0] * width)
    l_ankle_y  = int(LEFT_ANKLE[1] * height)
    r_ankle_x  = int(RIGHT_ANKLE[0] * width)
    r_ankle_y  = int(RIGHT_ANKLE[1] * height)
    c_ankle_x  = int(CENTER_ANKLE[0] * width)
    c_ankle_y  = int(CENTER_ANKLE[1] * height)

    l_heel_x   = int(LEFT_HEEL[0] * width)
    l_heel_y   = int(LEFT_HEEL[1] * height)
    r_heel_x   = int(RIGHT_HEEL[0] * width)
    r_heel_y   = int(RIGHT_HEEL[1] * height)
    c_heel_x   = int(CENTER_HEEL[0] * width)
    c_heel_y   = int(CENTER_HEEL[1] * height)

    l_foot_x   = int(LEFT_FOOT_INDEX[0] * width)
    l_foot_y   = int(LEFT_FOOT_INDEX[1] * height)
    r_foot_x   = int(RIGHT_FOOT_INDEX[0] * width)
    r_foot_y   = int(RIGHT_FOOT_INDEX[1] * height)
    c_foot_x   = int(CENTER_FOOT_INDEX[0] * width)
    c_foot_y   = int(CENTER_FOOT_INDEX[1] * height)
    #-------------------------
    # Analyze Torso
    #-------------------------
    l_torso_cross       = findDistance( l_shldr_x, l_shldr_y, r_hip_x, r_hip_y)
    r_torso_cross       = findDistance( r_shldr_x, r_shldr_y, l_hip_x, l_hip_y)
    center_torso_hip = (int((l_hip_x + r_hip_x)/ 2), int((l_hip_y + r_hip_y) /2))
    center_torso_shldr = (int((l_shldr_x + r_shldr_x) / 2), int((l_shldr_y + r_shldr_y) /2))
    center_torso = ((int((center_torso_hip[0] + center_torso_shldr[0])/2), int((center_torso_hip[1] + center_torso_shldr[1])/2)))
    torso_width = np.linalg.norm(center_torso_shldr[0] - center_torso_hip[0])
    TORSO_WIDTH = np.linalg.norm(CENTER_SHOULDER[0] - CENTER_HIP[0])
    TORSO_WIDTH_XPND = TORSO_WIDTH * width
    torso_height = np.linalg.norm(center_torso_shldr[1] - center_torso_hip[1])
    print("TORSO_WIDTH", TORSO_WIDTH, "TORSO_WIDTH_XPND", TORSO_WIDTH_XPND, "torso_width", torso_width, "torso_height", torso_height)
    print(80 * '-')
    print("l_torso_cross", l_torso_cross, "r_torso_cross", r_torso_cross,"center_torso", center_torso)

    #=======================================
    # MIDPOINTS
    #=======================================
    #-------------------------
    # Breast
    #-------------------------
    BREAST_MIDPOINT = ((LEFT_BREAST[0] + RIGHT_BREAST[0])/2,(LEFT_BREAST[1] + RIGHT_BREAST[1])/2)
    #-------------------------
    # Mouth
    #-------------------------
    mouth_midpoint = [(float(LEFT_MOUTH[0]) + float(RIGHT_MOUTH[0]) )/2
                       , (float(LEFT_MOUTH[1]) + float(RIGHT_MOUTH[1]))/2]
    SHOULDER_MIDPOINT = [(float(LEFT_SHOULDER[0]) + float(RIGHT_SHOULDER[0]) )/2
                       , (float(LEFT_SHOULDER[1]) + float(RIGHT_SHOULDER[1]))/2]
    HIP_MIDPOINT = [(float(LEFT_HIP[0]) + float(RIGHT_HIP[0]) )/2
                       , (float(LEFT_HIP[1]) + float(RIGHT_HIP[1]))/2]
    SOLAR_PLEXUS_MIDPOINT = [(float(LEFT_SOLAR_PLEXUS[0]) + float(RIGHT_SOLAR_PLEXUS[0]) )/2
                       , (float(LEFT_SOLAR_PLEXUS[1]) + float(RIGHT_SOLAR_PLEXUS[1]))/2]
    LEFT_CROSS = [(float(LEFT_SHOULDER[0]) + float(RIGHT_HIP[0])) /2
                         ,(float(LEFT_SHOULDER[1]) + float(RIGHT_HIP[1]))/2]


    RIGHT_CROSS = [(float(RIGHT_SHOULDER[0]) + float(LEFT_HIP[0])) /2
                         ,(float(RIGHT_SHOULDER[1]) + float(LEFT_HIP[1]))/2]

    CENTER_CROSS = [(LEFT_CROSS[0] + RIGHT_CROSS[0])/2, (LEFT_CROSS[1] + RIGHT_CROSS[1])/2]
    left_cross_x  = int(LEFT_CROSS[0] * width)
    left_cross_y  = int(LEFT_CROSS[1] *height)
    right_cross_x = int(RIGHT_CROSS[0] * width)
    right_cross_y = int(RIGHT_CROSS[1] *height)
    c_cross_x     = int(CENTER_CROSS[0] * width)
    c_cross_y     = int(CENTER_CROSS[1]  * height)

    ##############################
    # RIGHT_CROSS and LEFT_CROSS are the midpoints offset from solar_plexus midpoint. Join to right_poly_top_pts, left_poly_top_pts
    ##############################
    right_poly_top_pts = np.array([[r_shldr_x, r_shldr_y],(tuple(np.multiply(HIP_MIDPOINT, [image.shape[1], image.shape[0]]).astype(int)))], np.int32)
    left_poly_top_pts = np.array([[l_shldr_x, l_shldr_y], (tuple(np.multiply(HIP_MIDPOINT,  [image.shape[1], image.shape[0]]).astype(int)))], np.int32)
    #-------------------------
    # Hips2Shoulders
    #-------------------------
    hip2shldr_width  =  int(round(((CENTER_HIP[0] * width) - (CENTER_SHOULDER[0] * width))))
    hip2shldr_height =  int(round(((CENTER_HIP[1] * height) -(CENTER_SHOULDER[1] * height))))
    #-------------------------
    # Calculate Head dimensions
    #-------------------------
    head_width = int(round((shldr_length * .5)))
    head_length = int(round((hip2shldr_height * .25)))
    print("head_width", head_width, "head_length", head_length, "hip2shldr_width", hip2shldr_width, "hip2shldr_height", hip2shldr_height)
    #--------------------#
    # Solar Plexus - which side is compressed (carrying load) ?
    #--------------------#
    l_solar_length = findDistance(l_shldr_x, l_shldr_y, l_hip_x, l_hip_y)
    r_solar_length = findDistance(r_shldr_x, r_shldr_y, r_hip_x, r_hip_y)
    print("Solar Lengths: l_solar_length", l_solar_length, "r_solar_length", r_solar_length, "LEFT_HIP", LEFT_HIP, "RIGHT_HIP", RIGHT_HIP)
    #----------------------------------------#
    # Determine Load-Bearing Leg
    #----------------------------------------#

    #Notes: The straigher leg carries the load, generally
        # * Look to knee angle, which one is smaller angle?
        # * Which leg has longer hip to (ankle or foot_index)?
        # * take line from face center and extend to floor. 
        #       which foot is closer to center of gravity?
        # *** NOT IF YOU ARE IN A MARTIAL ARTS POSE! 

    l_knee_angle = findAngle(l_knee_x, l_knee_y, l_hip_x, l_hip_y ) #Askew20230823
    r_knee_angle = findAngle(r_knee_x, r_knee_y, r_hip_x, r_hip_y)  #Askew20230823
    print("l_knee_angle", l_knee_angle, "r_knee_angle", r_knee_angle)

    l_hip_length = findDistance(l_hip_x, SOLAR_PLEXUS_MIDPOINT[0] * width, l_hip_y, SOLAR_PLEXUS_MIDPOINT[1] * height)
    r_hip_length = findDistance(r_hip_x, SOLAR_PLEXUS_MIDPOINT[0] * width, r_hip_y, SOLAR_PLEXUS_MIDPOINT[1] * height)
    print("l_hip_length", l_hip_length, "r_hip_length", r_hip_length)
    #-------------------------
    # Determine hip to foot index - longer is load bearing leg
    #-------------------------
    l_hip2foot_length = findDistance(l_hip_x, l_hip_y, LEFT_HEEL[0] * width, LEFT_HEEL[1] * height)
    r_hip2foot_length = findDistance(r_hip_x, r_hip_y, RIGHT_HEEL[0] * width, RIGHT_HEEL[1] * height)
    print("l_hip2foot_length", l_hip2foot_length, "r_hip2foot_length", r_hip2foot_length)

    
    
    #----------------------------------------#
    # Determine Center line of Gravity
    #----------------------------------------#
    # Create alternative centerline perpendicular to c_head_x, c_head_y
    head_center_x = c_ear_x
    head_center_y = c_ear_y
    # which foot is closer to center of gravity?
    l_heel2center = findDistance(l_heel_x, l_heel_y, c_hip_x, c_hip_y)
    r_heel2center = findDistance(r_heel_x, r_heel_y, c_hip_x, c_hip_y)

    l_foot_index2center = findDistance(l_foot_x, l_foot_y, c_hip_x, c_hip_y)
    r_foot_index2center = findDistance(r_foot_x, r_foot_y, c_hip_x, c_hip_y) # zzzzz
    if r_foot_index2center > l_foot_index2center:# l_heel2center > r_heel2center:
        print("left heel closet")
    elif  l_foot_index2center > r_foot_index2center: #r_heel2center > l_heel2center:
        print("right heel closet")
    else:
        print("heels are equal to center hip")
    #---------------------------------
    # Determine Center of Body
    #---------------------------------
    center_line_length = findDistance(head_center_x, head_center_y, c_heel_x, c_heel_y)
    center_point  = (c_hip_x, c_hip_y)
    print("center_point:", center_point)
    #---------------------------------
    # Get angle of centerline to neck - same angle to offset breastline in gesture
    #---------------------------------
    # Create alternative centerline perpendicular to c_head_x, c_head_y
    # Get the lowest point on Gesture outline plot generated by OpenCV
    # center_line_length = findDistance(head_center_x, head_center_y, c_heel_x, c_heel_y)
   
    l_shldr2head = findDistance(head_center_x, head_center_y, l_shldr_x, l_shldr_y)
    r_shldr2head = findDistance(head_center_x, head_center_y, r_shldr_x, r_shldr_y)
    l_head_angle = calculate_angle((c_ear_x, c_ear_y),(c_shldr_x, c_shldr_y),(l_shldr_x, l_shldr_y))
    r_head_angle = calculate_angle((c_ear_x, c_ear_y),(c_shldr_x, c_shldr_y),(r_shldr_x, r_shldr_y))
    print("l_head_angle", l_head_angle, "r_head_angle", r_head_angle,  "l_shldr2head", l_shldr2head, "r_shldr2head", r_shldr2head)
    if l_shldr2head > r_shldr2head:
        if l_solar_length > r_solar_length:
            SHOULDER_HIGHER = "LEFT"
            print("LEFT SHOULDER Higher." "l_shldr2head", l_shldr2head, "r_shldr2head", r_shldr2head) #Askew20230818
        elif r_solar_length > l_solar_length:
            SHOULDER_HIGHER = "RIGHT"
            print("RIGHT SHOULDER Higher." "l_shldr2head", l_shldr2head, "r_shldr2head", r_shldr2head) #Askew20230818
    elif r_shldr2head > l_shldr2head:
        if l_solar_length > r_solar_length:
            SHOULDER_HIGHER ="LEFT"
            print("LEFT SHOULDER Higher." "l_shldr2head", l_shldr2head, "r_shldr2head", r_shldr2head) #Askew20230818
        elif r_solar_length > l_solar_length:
            SHOULDER_HIGHER ="RIGHT"
            print("RIGHT SHOULDER Higher." "l_shldr2head", l_shldr2head, "r_shldr2head", r_shldr2head) #Askew20230818       
    else:
        SHOULDER_HIGHER = "EQUAL"
        print("Shoulders are equal in distance to head: l_shldr2head", l_shldr2head, "r_shldr2head", r_shldr2head) #Askew20230818

#===========================================#
# CALCULATE DISTANCES and ANGLES
#============================================#
    #---------------------------------
    # ROTATION
    #---------------------------------
    #Calculate angle
    #-------------------------
    LEFT_ELBOW_angle = calculate_angle(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
    RIGHT_ELBOW_angle = calculate_angle(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
    #-------------------------
    #Calculate angle
    #-------------------------
    LEFT_SHOULDER_angle = calculate_angle(LEFT_ELBOW, LEFT_HIP, LEFT_SHOULDER)
    LEFT_SHOULDER_angle_abs = absAngle(LEFT_SHOULDER_angle)
    RIGHT_SHOULDER_angle = calculate_angle( RIGHT_HIP, RIGHT_ELBOW, RIGHT_SHOULDER)
    RIGHT_SHOULDER_angle_abs = absAngle(RIGHT_SHOULDER_angle)
     #-------------------------------
    # Extend Navel Line
    #-------------------------------
    l_angle_navel   = findAngle(l_shldr_x, l_shldr_y, l_navel_x, l_navel_y )
    l_angle_navel_abs = absAngle(l_angle_navel)
    r_angle_navel   = findAngle(r_shldr_x, r_shldr_y, r_navel_x, r_navel_y )
    r_angle_navel_abs = absAngle(r_angle_navel)
    navel_length = findDistance(l_navel_x, l_navel_y, r_navel_x, r_navel_y)/4
    l_navel_to_center_angle = calculate_angle(LEFT_NAVEL, CENTER_NAVEL, CENTER_HIP)
    #-------------------------
    # Navel midpoint
    #-------------------------
    l_navel_to_center_angle_abs = absAngle(l_navel_to_center_angle)
    r_navel_to_center_angle = calculate_angle(RIGHT_NAVEL, CENTER_NAVEL, CENTER_HIP)
    r_navel_to_center_angle_abs = absAngle(r_navel_to_center_angle)
    #-------------------------
    #Rib Cage midpoint
    #-------------------------
    l_rib_cage_to_center_angle = calculate_angle(LEFT_RIB_CAGE, CENTER_RIB_CAGE, CENTER_SHOULDER)
    rib_cage_length = findDistance(l_rib_cage_x, l_rib_cage_y, r_rib_cage_x, r_rib_cage_y)/4
    l_angle_rib_cage = findAngle(l_hip_x, l_hip_y, l_rib_cage_x, l_rib_cage_y)
    l_angle_rib_cage_abs = absAngle(l_angle_rib_cage)
    r_rib_cage_to_center_angle = calculate_angle(RIGHT_RIB_CAGE, CENTER_RIB_CAGE, CENTER_SHOULDER)
    r_angle_rib_cage = findAngle(r_hip_x, r_hip_y, r_rib_cage_x, r_rib_cage_y)
    r_angle_rib_cage_abs = absAngle(r_angle_rib_cage)

    c_rib_cage_angle = findAngle(l_rib_cage_x, l_rib_cage_y, r_rib_cage_x, r_rib_cage_y)
    c_rib_cage_angle_abs = absAngle(c_rib_cage_angle)
    #-------------------------------------------
    # HIP Angles
    #-------------------------------------------
    l_hip_to_center_angle = calculate_angle(LEFT_HIP, CENTER_HIP, CENTER_SOLAR_PLEXUS) #Askew20230829
    l_hip_to_center_angle_n = calculate_angle_numpy(LEFT_HIP, CENTER_HIP, CENTER_SOLAR_PLEXUS) #Askew20230829
    l_hip_to_center_angle_abs = absAngle(l_hip_to_center_angle)
    r_hip_to_center_angle = calculate_angle(RIGHT_HIP, CENTER_HIP, CENTER_SOLAR_PLEXUS) #Askew20230829
    r_hip_to_center_angle_n = calculate_angle_numpy(RIGHT_HIP, CENTER_HIP, CENTER_SOLAR_PLEXUS) #Askew20230829
    r_hip_to_center_angle_abs = absAngle(r_hip_to_center_angle)

    l_hip_angle = findAngle( l_shldr_x, l_shldr_y,  l_hip_x, l_hip_y)
    l_hip_angle_abs = absAngle(l_hip_angle)
    r_hip_angle = findAngle( r_shldr_x, r_shldr_y,  r_hip_x, r_hip_y)
    r_hip_angle_abs = absAngle(r_hip_angle)
    c_hip_angle = findAngle(l_hip_x, l_hip_y, r_hip_x, r_hip_y)
    c_hip_angle_abs = absAngle(c_hip_angle)
    #-------------------------------------------
    # Breasts
    #-------------------------------------------
    # l_breast_angle = calculate_angle( [l_solar_x, l_solar_y], [l_breast_x, l_breast_y], [l_shldr_x, l_shldr_y])
    l_breast_angle = calculate_angle([l_shldr_x, l_shldr_y], [l_breast_x, l_breast_y], [r_breast_x, r_breast_y])
    l_breast_angle_np = calculate_angle_numpy([l_shldr_x, l_shldr_y], [l_breast_x, l_breast_y], [r_breast_x, r_breast_y])
    l_breast_angle_abs = absAngle(l_breast_angle)
    # r_breast_angle = calculate_angle( [r_solar_x, r_solar_y], [r_breast_x, r_breast_y], [r_shldr_x, r_shldr_y])
    r_breast_angle = calculate_angle([r_shldr_x, r_shldr_y], [r_breast_x, r_breast_y], [l_breast_x, l_breast_y])
    r_breast_angle_np = calculate_angle_numpy([r_shldr_x, r_shldr_y], [r_breast_x, r_breast_y], [l_breast_x, l_breast_y])
    r_breast_angle_abs = absAngle(r_breast_angle)
    c_breast_angle  = (l_breast_angle_abs + r_breast_angle_abs)/2
    c_breast_angle_abs = absAngle(c_breast_angle)
    breast_length = width/findDistance(l_breast_x, l_breast_y, r_breast_x, r_breast_y)
    c_breast_angle_delta = angleDelta(l_breast_angle, r_breast_angle)
    #----------------------------
    # Calculate BENDS
    #----------------------------
    torso_bend_length = abs(int(abs(l_hip_x - l_shldr_x)) - int(abs(r_hip_x - r_shldr_x)))
    print(" torso_bend_length",  torso_bend_length)
    shldr_bend  = angleDelta(LEFT_SHOULDER_angle_abs, RIGHT_SHOULDER_angle_abs)
    rib_cage_bend = angleDelta(l_rib_cage_to_center_angle, r_rib_cage_to_center_angle)
    navel_bend = angleDelta(l_angle_rib_cage_abs, r_angle_rib_cage_abs)
    hip_bend = angleDelta(l_hip_to_center_angle_abs, r_hip_to_center_angle_abs)
    total_bend = (shldr_bend + rib_cage_bend + navel_bend + hip_bend)
    hip_angle_delta = angleDelta(l_hip_angle_abs, r_hip_angle_abs)
    l_hip2breast_angle = angleDelta(l_hip_angle_abs, l_breast_angle_abs)
    r_hip2breast_angle = angleDelta(r_hip_angle_abs, r_breast_angle_abs)
    l_shldr2hip_angle = angle_between((l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y))
    r_shldr2hip_angle = angle_between((r_hip_x, r_hip_y), (r_shldr_x, r_shldr_y))
    print(80 * '-')
    print("BEND_ANGLES: l_shldr2hip_angle", l_shldr2hip_angle, "r_shldr2hip_angle", r_shldr2hip_angle)
    print(80 * '-')
    print("BENDS: torso_bend_length", torso_bend_length, "shldr_bend", shldr_bend, "rib_cage_bend", rib_cage_bend, "navel_bend", navel_bend,"Hip Bend", hip_bend,  )
    print("l_angle_rib_cage", l_angle_rib_cage, "r_angle_rib_cage", r_angle_rib_cage, "l_angle_navel", l_angle_navel, "r_angle_navel", r_angle_navel)
    print("navel_length", navel_length, "torso_width", torso_width,"l_angle_navel", l_angle_navel, "r_angle_navel", r_angle_navel)
    #-------------------------------------------
    # TORSO ROTATION: Angle Breasts to Hip
    #-------------------------------------------
    torso_rotation = angleDelta(c_shldr_angle_abs, c_hip_angle_abs)#c_breast_angle)
    if torso_rotation > 0:
        ROTATION_DIRECTION = ROTATION_POSITIVE
    elif torso_rotation < 0:
         ROTATION_DIRECTION = ROTATION_NEGATIVE
    else:
        ROTATION_DIRECTION = "NO ROTATION"
    #-------------------------
    # Once rotation angle established, get distance between angle points
    #-------------------------
    c_shldr_2_c_breast = findDistance(c_shldr_x, c_shldr_y, c_breast_x, c_breast_y)
    c_breast_2_c_solar = findDistance(c_breast_x, c_breast_y, c_solar_x, c_solar_y)
    c_solar_2_c_navel = findDistance(c_solar_x, c_solar_y, c_navel_x, c_navel_y)
    c_navel_2_c_hip    = findDistance(c_navel_x, c_navel_y, c_hip_x, c_hip_y)
    rotate_length = (c_hip_x -c_shldr_x ) #Askew20230824 zzz
    print("Rotation length total", rotate_length, "c_shldr_2_c_breast", c_shldr_2_c_breast, "c_breast_2_c_solar", c_breast_2_c_solar, "c_solar_2_c_navel", c_solar_2_c_navel,  "c_navel_2_c_hip", c_navel_2_c_hip)
    #-------------------------
    # Get the new point based on rotation_length
    #-------------------------
    shldr_angle_offset_rect = point_pos_rect(SHOULDER_MIDPOINT, shldr_length, shldr_bend)
    print("shldr_angle_offset_rect", shldr_angle_offset_rect)
    solar_rotation_offset = point_pos(SOLAR_PLEXUS_MIDPOINT[0], SOLAR_PLEXUS_MIDPOINT[1], rotate_length, torso_rotation)
    solar_rotation_offset_rect = point_pos_rect(SOLAR_PLEXUS_MIDPOINT, rotate_length, torso_rotation)
    torso_rotation_offset = point_pos(center_torso[0], center_torso[1], rotate_length, torso_rotation)
    torso_rotation_offset_rect = point_pos_rect(center_torso, rotate_length, torso_rotation)
    if torso_rotation < 0:
        torso_rotation_offset_rect_points = (int(torso_rotation_offset_rect[0] + torso_width/2)) #xxx Askew20230825
    else:
        torso_rotation_offset_rect_points = (int(torso_rotation_offset_rect[0] - torso_width/2)) 
    print("solar_rotation_offset_rect", solar_rotation_offset_rect,"solar_rotation_offset", solar_rotation_offset, "SOLAR_PLEXUS_MIDPOINT", SOLAR_PLEXUS_MIDPOINT)
    print("torso_rotation_offset_rect", torso_rotation_offset_rect,"torso_rotation_offset", torso_rotation_offset, "center_torso", center_torso)
    
    # Not needed --> cv2.circle(image, (int(torso_rotation_offset_rect[0]), int(torso_rotation_offset_rect[1])), 5, dark_blue, 7)
    # Not needed --> cv2.circle(image, (int(torso_rotation_offset[0]), int(torso_rotation_offset[1])), 6, yellow, 8)
    #-------------------------
    # Continue with ROTATION determination
    #-------------------------
    if l_torso_cross > r_torso_cross:
        BEND_DIRECTION = (BEND_RIGHT, torso_bend_length)
    elif r_torso_cross > l_torso_cross:
        BEND_DIRECTION = (BEND_LEFT, torso_bend_length)
    else:
        BEND_DIRECTION=("NO BENDING", hip_bend)
    print("BEND", BEND_DIRECTION, "TORSO ROTATION", torso_rotation, ROTATION_DIRECTION, "ROTATIONAL LENGTH", rotate_length, )
    print(80 * '-')
    #-------------------------
    # Face Angles
    #-------------------------
    c_face_angle = findAngle(c_ear_x, c_ear_y, SHOULDER_MIDPOINT[0], SHOULDER_MIDPOINT[1])
    print("c_face_angle", c_face_angle)
    #-------------------------
    # Verify Stance and Posture
    #-------------------------
    print( 80 * '=')
    print("HIP INFO")
    print( 80 * '=')
    
    print("LEFT_HIP", LEFT_HIP, "RIGHT_HIP", RIGHT_HIP, "CENTER_HIP", CENTER_HIP)
    if bool(LEFT_HIP) & bool(RIGHT_HIP):
       print("BOTH HIPS FOUND")
    print("Lenth of Rotation from Hip to Shoulder:", rotate_length)
    print("torso_rotation", torso_rotation, ROTATION_DIRECTION, "(l_hip2breast_angle - r_hip2breast_angle)", (l_hip2breast_angle + r_hip2breast_angle), "l_hip2breast_angle", l_hip2breast_angle,  "r_hip2breast_angle", r_hip2breast_angle)
    print( "HIP :torso_bend_length", torso_bend_length )
    print("hip_bend", hip_bend)
    print("l_shldr2hip_angle", l_shldr2hip_angle, "r_shldr2hip_angle", r_shldr2hip_angle)
    print("#-------------")
    print( "Hips to Navel info")
    print("#-------------")
    print("CENTER HIP to navel info l_navel_to_center_angle", l_navel_to_center_angle, "r_navel_to_center_angle", r_navel_to_center_angle)
    print("l_navel_to_center_angle_abs", l_navel_to_center_angle_abs)
    print("r_navel_to_center_angle_abs", r_navel_to_center_angle_abs)
    print("c_navel_2_c_hip", c_navel_2_c_hip)
    print("From Both Hips to Ribcage Angles: l_angle_rib_cage", l_angle_rib_cage, "r_angle_rib_cage", r_angle_rib_cage)
    print("l_hip_angle",l_hip_angle)
    print("l_hip_angle_abs",l_hip_angle_abs)
    print("r_hip_angle", r_hip_angle)
    print("r_hip_angle_abs", r_hip_angle_abs)
    print("c_hip_angle", c_hip_angle)
    print("c_hip_angle_abs", c_hip_angle_abs)
    print("l_hip_to_center_angle", l_hip_to_center_angle)
    print("l_hip_to_center_angle_n", l_hip_to_center_angle_n)
    print("l_hip_to_center_angle_abs", l_hip_to_center_angle_abs )
    print("r_hip_to_center_angle", r_hip_to_center_angle)
    print("r_hip_to_center_angle_n", r_hip_to_center_angle_n)
    print("r_hip_to_center_angle_abs", r_hip_to_center_angle_abs)

    print("l_hip_angle", l_hip_angle)
    print("l_hip_angle_abs", l_hip_angle_abs)
    print("r_hip_angle", r_hip_angle)
    print("r_hip_angle_abs", r_hip_angle_abs)
    print("c_hip_angle", c_hip_angle)
    print("c_hip_angle_abs", c_hip_angle_abs)
   
    print("l_hip_to_center_angle", l_hip_to_center_angle, "l_hip_to_center_angle_n", l_hip_to_center_angle_n, "l_hip_to_center_angle_abs", l_hip_to_center_angle_abs)
    print("r_hip_to_center_angle", r_hip_to_center_angle, "r_hip_to_center_angle_n", r_hip_to_center_angle_n, "r_hip_to_center_angle_abs", r_hip_to_center_angle_abs)
    print("RATIO Hip to Torso. torso_width =", torso_width, "hip_length", hip_length, "l_hip_length", l_hip_length, "hip_angle_delta", hip_angle_delta,  "l_hip_angle_abs", l_hip_angle_abs, "l_hip_angle", l_hip_angle, "l_hip_x", l_hip_x, "l_hip_y", l_hip_y)
    print("RATIO Hip to Torso. torso_width =", torso_width, "hip_length", hip_length, "r_hip_length", r_hip_length, "hip_angle_delta", hip_angle_delta, "r_hip_angle_abs", r_hip_angle_abs, "r_hip_angle", r_hip_angle, "r_hip_x", r_hip_x, "r_hip_y", r_hip_y)
    print("hip_length" , hip_length, "l_hip_length", l_hip_length, "hip_angle_delta", hip_angle_delta,  "l_hip_angle_abs", l_hip_angle_abs, "l_hip_angle", l_hip_angle, "l_hip_x", l_hip_x, "l_hip_y", l_hip_y)
    print("hip_length", hip_length, "r_hip_length", r_hip_length, "hip_angle_delta", hip_angle_delta, "r_hip_angle_abs", r_hip_angle_abs, "r_hip_angle", r_hip_angle, "r_hip_x", r_hip_x, "r_hip_y", r_hip_y)
    #-------------------------------------------
    # Perspective: Calculate visible hips to torso height
    #-------------------------------------------
    print("torso_height", torso_height, "torso_height * HIP_W_2_TORSO", (torso_height * HIP_W_2_TORSO))
    if hip_length < (torso_height * HIP_W_2_TORSO):
        print("Perspective: Hips Foreshortened")
    else:
        print("Perspective Hips NOT Foreshortened")
    #------------------------------------------#
    # Solor Plexus Angles
    #------------------------------------------#
    l_solar_plexus_angle = calculate_angle(LEFT_SOLAR_PLEXUS, LEFT_KNEE, LEFT_HIP)
    r_solar_plexus_angle = calculate_angle(RIGHT_SOLAR_PLEXUS, RIGHT_KNEE, RIGHT_HIP)
   
    solar_plexus_length  = findDistance(LEFT_SOLAR_PLEXUS[0]* width, LEFT_SOLAR_PLEXUS[1]* height, RIGHT_SOLAR_PLEXUS[0]* width, RIGHT_SOLAR_PLEXUS[1]* height)
    print("solar_plexus_length", solar_plexus_length, "torso_width", torso_width)
    #------------------------------------------#
    # Try Shoulders
    #------------------------------------------#
    l_shldr_angle = findAngle( l_shldr_x, l_shldr_y, l_solar_x, l_solar_y,)# r_shldr_y)
    r_shldr_angle = findAngle( r_shldr_x, r_shldr_y, r_solar_x, r_solar_y,)
    shldr_length = width/findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y) 

    #-------------------------------
    ## Get distance node to mouth center
    #-------------------------------
    mouth_angle = findAngle(l_shldr_x, l_shldr_y, l_mouth_x, right_mouth_y)
    nose2mouth_length  = findDistance(NOSE[0]* width, NOSE[1]* height, CENTER_MOUTH[0]* width, CENTER_MOUTH[1]* height)
    angle_calc_nose2shldr_mid = calculate_angle( [c_ear_x, c_ear_y], mouth_midpoint, [(SHOULDER_MIDPOINT[0] * width), (SHOULDER_MIDPOINT[1] * height)])
    angle_ears2mouth_shlders = findAngle(c_ear_x, c_ear_y, c_shldr_x, c_shldr_y) #c_mouth_x, c_mouth_y)
    print("ANGLES: mouth_angle", mouth_angle, "nose2mouth_length", nose2mouth_length, "angle_calc_nose2shldr_mid",angle_calc_nose2shldr_mid,"angle_ears2mouth_shlders", angle_ears2mouth_shlders)
    ##
    ## Inclinations
    ##
    # Get Neck and Torso
    l_neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
    r_neck_inclination = findAngle(r_shldr_x, r_shldr_y, r_ear_x, r_ear_y)

    l_torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)
    r_torso_inclination = findAngle(r_hip_x, r_hip_y, r_shldr_x, r_shldr_y)
    angle_text_string = 'Neck : ' + str(int(l_neck_inclination)) + '  Torso : ' + str(int(l_torso_inclination))


    #------------------------------
    # Plot curve
    #------------------------------
    
    my_array = []
    my_array = np.array([[l_solar_x + (np.pi/8 * rotate_length * torso_rotation), l_solar_y],(tuple(np.multiply(SOLAR_PLEXUS_MIDPOINT, [image.shape[1], image.shape[0]]).astype(int)))], np.int32)
    print("my_array", my_array)
    my_array = np.array(my_array).astype(np.int32)
    my_array = my_array.reshape(-1,1,2)
     
    #=========================================
    # RENDER OpenCV's model-based supplied GESTURES
    #=========================================
        # Render detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
    mp_drawing.DrawingSpec(color=cv2_pts_olive_green, thickness=2, circle_radius=2), # 245,117,66
    mp_drawing.DrawingSpec(color=cv2_lines_royal_blue, thickness=2, circle_radius=2) # 245,66,230
    )
    ##########################################
    # Render (Visualize) Drawings 
    ##########################################
    #----------------------------------------#
    # Plot Center of Body
    #----------------------------------------#
    # DEBUG only --> cv2.circle(image, (center_point[0], center_point[1]), 5, red, 8)
    #----------------------------------------#
    # Load Bearing leg
    #----------------------------------------#
    if ((l_solar_length > r_solar_length) & (r_hip2foot_length > l_hip2foot_length) & (l_hip_angle_abs > r_hip_angle_abs)):
    #if ((l_solar_length > r_solar_length) & (r_hip2foot_length > l_hip2foot_length) & (l_hip_to_center_angle_n > r_hip_to_center_angle_n)):
        #HIP_HIGHER = "RIGHT"
        visualize_right_side(image, head_center_x, head_center_y, c_shldr_x, c_shldr_y, c_breast_x, c_breast_y, center_torso, c_navel_x, c_navel_y, c_hip_x, c_hip_y, r_knee_x, r_knee_y , torso_width, torso_height, width, height, torso_rotation_offset_rect, center_line_offset_x, center_line_offset_y, center_line_green, center_line_color )
    elif ((SHOULDER_HIGHER == "LEFT") &(r_solar_length > l_solar_length) & (l_hip2foot_length > r_hip2foot_length) & (r_hip_angle_abs > l_hip_angle_abs)):
        HIP_HIGHER = "LEFT"
        visualize_left_size(image, head_center_x, head_center_y, c_shldr_x, c_shldr_y, c_breast_x, c_breast_y, center_torso, c_navel_x, c_navel_y, c_hip_x, c_hip_y, r_knee_x, r_knee_y , torso_width, torso_height, width, height, torso_rotation_offset_rect, center_line_offset_x, center_line_offset_y, center_line_green, center_line_color )
    else: 
        HIP_HIGHER = "EQUAL"
        print("SIDES Equal")
        if l_knee_angle > r_knee_angle:
            visualize_right_side(image, head_center_x, head_center_y, c_shldr_x, c_shldr_y, c_breast_x, c_breast_y, center_torso, c_navel_x, c_navel_y, c_hip_x, c_hip_y, r_knee_x, r_knee_y , torso_width, torso_height, width, height, torso_rotation_offset_rect, center_line_offset_x, center_line_offset_y, center_line_green, center_line_color )
        elif r_knee_angle > l_knee_angle:
            visualize_left_side(image, head_center_x, head_center_y, c_shldr_x, c_shldr_y, c_breast_x, c_breast_y, center_torso, c_navel_x, c_navel_y, c_hip_x, c_hip_y, r_knee_x, r_knee_y , torso_width, torso_height, width, height, torso_rotation_offset_rect, center_line_offset_x, center_line_offset_y, center_line_green, center_line_color )
        else:
            print("HEELS EVEN")
            print("Weight bears evenly both legs") #yyyyy
            if r_foot_index2center > l_foot_index2center: #c_heel_x:
                cv2.line(image, ((head_center_x -center_line_offset_x), (head_center_y - center_line_offset_y)), (l_foot_x, l_foot_y), center_line_color, 1)
            elif l_foot_index2center > r_foot_index2center: #c_heel_x:
                cv2.line(image, ((head_center_x -center_line_offset_x), (head_center_y - center_line_offset_y)), (r_foot_x, r_foot_y), center_line_color, 1)
            else:
                cv2.line(image, ((head_center_x -center_line_offset_x), (head_center_y - center_line_offset_y)), (c_knee_x, c_knee_y), center_line_color, 1)
            
    #----------------------------
    # Continue Plotting
    #----------------------------
    

    #Visualize angle
    cv2.putText(image, str(LEFT_ELBOW_angle),
            tuple(np.multiply(LEFT_ELBOW, [height, width]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX, 1, my_white, 2, cv2.LINE_4)
    cv2.putText(image, str(RIGHT_ELBOW_angle),
            tuple(np.multiply(RIGHT_ELBOW, [height, width]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX, 1, my_white, 2, cv2.LINE_4)

    #Visualize angle
    cv2.putText(image, str("LSA:" + str(LEFT_SHOULDER_angle)),
            #tuple(np.multiply(LEFT_SHOULDER, [LEFT_SHOULDER[0] * height, LEFT_SHOULDER[1] * width]).astype(int)),
            (int(LEFT_SHOULDER[0] * image.shape[1]), puttext_offset + int(LEFT_SHOULDER[1] * image.shape[0])),
            #(int(LEFT_SHOULDER[0]), int(LEFT_SHOULDER[1])), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.25, my_white, 1, cv2.LINE_4)
    cv2.putText(image, str("RSA:" + str(RIGHT_SHOULDER_angle)),
            #tuple(np.multiply(RIGHT_SHOULDER, [580, 1200]).astype(int)),
            (int(RIGHT_SHOULDER[0] * image.shape[1]),  puttext_offset +int(RIGHT_SHOULDER[1] * image.shape[0])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.25, my_white, 1, cv2.LINE_4)

    cv2.putText(image, angle_text_string, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, light_green, 1)
   
    #============================
    # Mouth Center and Extend
    #============================
    cv2.circle(image, tuple(np.multiply(CENTER_EAR, [width, height]).astype(int)), 2, orange, 2) #Askew20230812
    #-------------------------------
    # Extend Breast Line
    #-------------------------------
    #BREAST_MIDPOINT
    # DEBUG only --> cv2.circle(image, (c_breast_x, c_breast_y), 2, red_orange, 2) #Askew20230812
    # Breast line
    P2, P3 = left_extract_points(l_breast_x, l_breast_y, l_breast_angle, breast_length)
    R_P2, R_P3 = right_extract_points(r_breast_x, r_breast_y, r_breast_angle, breast_length)
    cv2.line(image, (P2 , P3), (R_P2, R_P3),  red_orange, contra_lines_width) #dark_blue, 2)

    #-------------------------------
    # Extend Navel Line
    #-------------------------------
    #navel_midpoint
    # DEBUG only --> cv2.circle(image, (c_navel_x, c_navel_y), 2, red_orange, 2) #Askew20230812
    if navel_length > torso_width:
        length = torso_width
    else:
        length = navel_length
    P2, P3 = left_extract_points(l_navel_x, l_navel_y, l_angle_navel, length)
    R_P2, R_P3 = right_extract_points(r_navel_x, r_navel_y, r_angle_navel, length)
    cv2.line(image, (P2 , P3), (R_P2, R_P3),  red_orange, contra_lines_width) #dark_blue, 2)

    # Box in pelvis 
    cv2.line(image, (l_hip_x  , l_hip_y), (P2, P3), dark_blue, 1)
    cv2.line(image, (r_hip_x  , r_hip_y), (R_P2, R_P3),  dark_blue, 1)
    
    #-------------------------------
    # Extend Rib Cage Line 
    #-------------------------------
    P2, P3 = left_extract_points(l_rib_cage_x, l_rib_cage_y, l_angle_rib_cage, rib_cage_length/2)
    R_P2, R_P3 = right_extract_points(r_rib_cage_x, r_rib_cage_y, r_angle_rib_cage, rib_cage_length/2)
    
    cv2.line(image, (P2 , P3), (R_P2, R_P3), dark_blue, 5) #contra_lines_width) #red_orange, contra_lines_width) #dark_blue, 2)
    
    #-------------------------------
    #Box in Ribs from Shoulder to Rib Cage
    #-------------------------------
    cv2.line(image, (l_shldr_x, l_shldr_y), (l_rib_cage_x, l_rib_cage_y), dark_blue, 5)
    cv2.line(image, (r_shldr_x, r_shldr_y), (r_rib_cage_x, r_rib_cage_y), dark_blue, 5)
   
    #-------------------------------
    # Extend nose2shoulder
    #-------------------------------
    length =nose2mouth_length
    P1 = ()
    P1 = (c_ear_x, c_ear_y) #nose_xy[0], nose_xy[1] )
    P2 = (0,0)
    P2 =  (int(round(P1[0]  + (length)  * -np.cos(angle_ears2mouth_shlders * np.pi / 180.0))))
    P3 =  (int(round(P1[1]  + (length) *  np.sin(angle_ears2mouth_shlders * np.pi / 180.0))))
    #
    R_P1 = (c_shldr_x, c_shldr_y) #(c_mouth_x, c_mouth_y ) #+ (length *2000))
    R_P2 = (int(round(R_P1[0] + (length) *  np.cos(angle_ears2mouth_shlders * np.pi / 180.0))))
    R_P3 = (int(round(R_P1[1] +  (length) * np.sin(angle_ears2mouth_shlders * np.pi / 180.0))))
    #DEBUG NO Value cv2.line(image, (P2 , P3), (R_P2, R_P3), orange, 5)
    #--------------------------------
    # Nose to Mouth
    #--------------------------------
    nose2mouth_length_distance  = findDistance(nose_xy[0], nose_xy[1], CENTER_MOUTH[0], CENTER_MOUTH[1]) 
    print("nose2mouth_length", nose2mouth_length, "nose2mouth_length_distance", nose2mouth_length_distance, "angle_ears2mouth_shlders", angle_ears2mouth_shlders, "angle_calc_nose2shldr_mid", angle_calc_nose2shldr_mid)
    #DEBUG NO Value cv2.line(image, (P2, P3), (R_P2, R_P3),  red_orange, 2)
    #---------------------------------
    # Determine total head size xxxx
    #---------------------------------
    P1 = (c_ear_x, c_ear_y) #nose_xy[0], nose_xy[1] )
    P2 = (0,0)
    P2 =  (int(round(P1[0]  + (.5 * head_width)  * -np.cos(angle_ears2mouth_shlders * np.pi / 180.0)))) - int(round(head_width))
    P3 =  (int(round(P1[1]  + (.5 * head_length) *  np.sin(angle_ears2mouth_shlders * np.pi / 180.0))))  - int(round(head_length))
    #
    R_P1 = (c_shldr_x, c_shldr_y) #(c_mouth_x, c_mouth_y ) #+ (length *2000))
    R_P2 = (int(round(R_P1[0] + (nose2mouth_length) *  np.cos(angle_ears2mouth_shlders * np.pi / 180.0))))
    R_P3 = (int(round(R_P1[1] +  (nose2mouth_length) * np.sin(angle_ears2mouth_shlders * np.pi / 180.0))))
  
    # Not needed for DEBUG --> cv2.line(image, (P2 , P3), (R_P2, R_P3),  dark_blue, 5)
    
    
    
    #============================zzzzzw
    # mouth to shoulder LINE
    #============================

    head_top = point_pos_rect(((nose_xy[0]- (((2*np.pi/2) *  shldr_angle_offset_rect[0]) + ((2 * np.pi/4) *head_width))), (nose_xy[1] - head_length)), (nose2mouth_length_distance/width*torso_width) ,shldr_bend)
    print("head_top", head_top)
    # Does not work --> cv2.circle(image, [int(head_top[0]), int(head_top[1])] ,2, red, 5)

    P2, P3 = left_extract_points(c_ear_x, c_ear_y, -c_face_angle, length)
    R_P2, R_P3 = right_extract_points((c_shldr_x - int(round(torso_rotation_offset_rect[0]/width))), (c_shldr_y), c_face_angle, length) #Askew20230826
    cv2.line(image, (P2, P3), (R_P2, R_P3),  center_line_green, 5)
    #============================
    # Extend Hips
    #============================
    P2, P3 = left_extract_points(l_hip_x, l_hip_y, l_hip_angle, shldr_length)
    R_P2, R_P3 = right_extract_points(r_hip_x, r_hip_y, r_hip_angle, shldr_length)
    cv2.line(image, (l_hip_x  , l_hip_y), (P2, P3), red_orange, 1) 
    cv2.line(image, (r_hip_x  , r_hip_y), (R_P2, R_P3),  red_orange, 1)
    cv2.line(image, (P2 , P3), (R_P2, R_P3),  red_orange, contra_lines_width)

    #============================
    # Extend Solar Plexus
    #============================
    length = width/findDistance(l_solar_x, l_solar_y, r_solar_x, r_solar_y) 
    P2, P3 = left_extract_points(l_solar_x, l_solar_y, l_solar_plexus_angle, solar_plexus_length)
    R_P2, R_P3 = right_extract_points(r_solar_x, r_solar_y, r_solar_plexus_angle, solar_plexus_length)
    #---------------------------
    # Suppress solar_plexus line
    #----------------------------
    #cv2.line(image, (P2 , P3), (R_P2, R_P3),  red_orange, contra_lines_width) #dark_blue, 2)
        
     #============================
    # Extend Shoulders
    #============================
    P2, P3 = left_extract_points(l_shldr_x, l_shldr_y, l_shldr_angle, shldr_length)
    R_P2, R_P3 = right_extract_points(r_shldr_x, r_shldr_y, r_shldr_angle, shldr_length)
    cv2.line(image, (P2 , P3), (R_P2, R_P3),  dark_blue, 5) 
    #=============================
    # Extend Center of Eyes       #Askew20230812
    #=============================

    #============================
    # Split body down middle
    #============================
    cv2.line(image, (r_shldr_x, r_shldr_y), (tuple(np.multiply(HIP_MIDPOINT, [image.shape[1], image.shape[0]]).astype(int))), (purple), 1)
    cv2.line(image, (l_shldr_x, l_shldr_y), (tuple(np.multiply(HIP_MIDPOINT, [image.shape[1], image.shape[0]]).astype(int))), (purple), 1)

    cv2.polylines(image, [right_poly_top_pts], False, yellow)
    cv2.polylines(image, [left_poly_top_pts], False, yellow)

  
################################
## Experimental Delauney triangulation
################################
    my_array = np.array([(c_cross_x, c_cross_y) , (r_shldr_x, r_shldr_y),  (l_shldr_x, l_shldr_y)])
    for x,y in my_array:
        points.append((int(x), int(y)))
   # Insert points into subdiv
    for p in points :
        subdiv.insert(p)
    # Draw delaunay triangles
    draw_delaunay( img, subdiv, (255, 255, 255) );

    # Draw points
    for p in points :
        draw_point(img, p, (0,0,255))

    # Show results
    cv2.imshow(win_delaunay,img)
    cv2.imwrite('image_scanned.jpg', image)
    # cv2.imwrite(directory + '/' + split_tup[0] + '_scanned' + split_tup[1], image)
    while True:
        cv2.imshow('Mediapipe Feed', image)
        if cv2.waitKey(10) & 0xff ==ord('q'):
           break
    #Show image
        
cap.release()
cv2.destroyAllWindows()
