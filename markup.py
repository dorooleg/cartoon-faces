import cv2
import copy
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os

down = False
points = []
update_points = []

def draw_points():
    global ix, iy, points, new_img, down
    new_img = copy.deepcopy(img)
    cv2.imshow('image', new_img)
    for (i, (x, y)) in enumerate(points):
        cv2.circle(new_img, (x, y), 5, (255, 0, 0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(new_img, str(i), (x, y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def update_points_(x, y):
    global points, update_points
    for (ux, uy) in update_points:
        for i in range(len(points)):
            points[i] = (x, y) if points[i][0] == ux and points[i][1] == uy else points[i]
    update_points = [(x, y)]
    draw_points()

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,points,new_img,down,update_points
    if event == cv2.EVENT_LBUTTONDOWN:
        down = True
        update_points = [v for v in points if pow(x - v[0], 2) + pow(y - v[1], 2) <= 25]
    elif event == cv2.EVENT_LBUTTONUP:
        update_points = []
        down = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if down == True:
            update_points_(x, y)
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        points.append((x, y))
        draw_points()
    elif event == cv2.EVENT_RBUTTONDBLCLK:
        points = list(filter(lambda v: pow(x - v[0], 2) + pow(y - v[1], 2) >= 25, points))
        draw_points()

Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file

# Create a black image, a window and bind the function to window
img = cv2.imread(filename, -1)
new_img = copy.deepcopy(img)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',new_img)
    k = cv2.waitKey(20) & 0xFF
    if k == ord('q'):
        break
    if k == ord('a'):
        with open(os.path.splitext(filename)[0]+'.csv', 'w') as file:
            mylist_x = [str(x) for (x, _) in points]
            mylist_y = [str(x) for (_, x) in points]
            file.write(", ".join(mylist_x))
            file.write("\n")
            file.write(", ".join(mylist_y))


cv2.destroyAllWindows()