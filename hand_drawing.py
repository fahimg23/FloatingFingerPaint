import cv2
import numpy as np
import Tkinter as tk
from math import sqrt

cap = cv2.VideoCapture(0)

#tkinter gui root
root = tk.Tk()

frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
canv_width = frame_width
canv_height = frame_height
canv = tk.Canvas(root, width = canv_width, height = canv_height)
canv.pack()

circleRad = tk.IntVar()
circle_rad_slider = tk.Scale(root, from_ = 1, to = 20, label = "Brush Width", variable = circleRad)
circle_rad_slider.pack(anchor = tk.SE)

# allow user to change brush color by hovering over rectangles of those colors
rectangle_height = 25 # 50 pixels
rectangle_width = 100
rectangle_left_edge_pos = frame_width - rectangle_width #the rectangle right edge pos is equal to the frame_width
rectangle_padding = 100

blue_rect_xy = (0, 0)
blue_rect_x2y2 = (blue_rect_xy[0]+rectangle_width, blue_rect_xy[1] + rectangle_height) #blue_rect_xy_[1] is y coordinate of first rectangle vertex

green_rect_xy = (blue_rect_x2y2[0]+rectangle_padding, 0)
green_rect_x2y2 = (green_rect_xy[0]+rectangle_width, green_rect_xy[1] + rectangle_height)

red_rect_xy = (green_rect_x2y2[0]+rectangle_padding, 0)
red_rect_x2y2 = (red_rect_xy[0]+rectangle_width, red_rect_xy[1] + rectangle_height)

def draw_rects():
    canv.create_rectangle(blue_rect_xy, blue_rect_x2y2, fill = 'blue')
    canv.create_rectangle(green_rect_xy, green_rect_x2y2, fill = 'green')
    canv.create_rectangle(red_rect_xy, red_rect_x2y2, fill = 'red')

draw_rects()
    
def get_distance_between_points(point1, point2):

    x1 = point1[0]
    y1 = point1[1]

    x2 = point2[0]
    y2 = point2[1]

    x_dist_squared = (x2-x1)**2
    y_dist_squared = (y2-y1)**2
    
    dist = sqrt(x_dist_squared + y_dist_squared)
    return dist

# note, point_array_1 and point_array_2 are 2-D arrays, both having 2 arrays inside one large array
def get_distance_between_point_array(point_array_1, point_array_2):

    x1 = point_array_1[0]
    y1 = point_array_1[1]

    x2 = point_array_2[0]
    y2 = point_array_2[1]

    x_dist_squared = cv2.pow(cv2.subtract(x2, x1), 2)
    y_dist_squared = cv2.pow(cv2.subtract(y2, y1), 2)
    
    dist = cv2.sqrt(cv2.add(x_dist_squared, y_dist_squared))
    return dist

def find_farthest_point_from_contour_center(contour, defects, center):

    s = defects[:,0][:,0]

    x1 = np.array(contour[s][:,0][:,0], dtype = np.float)
    y1 = np.array(contour[s][:,0][:,1], dtype = np.float)

    dist = get_distance_between_point_array(center, (x1,y1))

    max_dist_index = np.argmax(dist)

    if max_dist_index < len(s):
        farthest_defect = s[max_dist_index]
        farthest_point = tuple(contour[farthest_defect][0])
        return farthest_point

# checks for overlap of circles (of same radius)
def check_circle_overlap(center1, center2, radius):

    #check if distance between the centers of the circles is less than the radius of one of the circles
    dist = get_distance_between_points(center1, center2)
    
    if(dist <= radius):
        return True

    return False

# rect_vertex1 should be top left vertex, and rect_vertex2 should be bottom right vertex
def check_point_inside_rectangle(point, rect_vertex1, rect_vertex2):

    if(point[0] > rect_vertex1[0] and point[0] < rect_vertex2[0] and point[1] > rect_vertex1[1] and point[1] < rect_vertex2[1]):
        return True

    return False

prev_farthest_point_from_center = None
default_color = '#000fff000' #pure green
color = default_color

first_run = True

while(cap.isOpened()):
    root.update()
    # capture frames
    ok, frame = cap.read()
    frame = cv2.flip(frame, 1) # flips image horizontally so when user draws, it draws in the same direction of their hand motion
    key = cv2.waitKey(1)

    # quit if q is pressed
    if(key == ord('q')):
        break

    # extract the region of interest (or remove the background)
    
    # convert to gray scale
    grayed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # blur the grayed image (removes gaussian noise from the image) 
    #(5,5) is the heigh and width of the gaussian kernel, both should be
    # positive and odd
    blurred_frame = cv2.GaussianBlur(grayed_frame,(35,35), 0)
    
    _,threshold = cv2.threshold(blurred_frame,100,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    cv2.imshow("Binary Mask/ Threshold", threshold)

    # get all the contours in the frame
    _, contours, _ = cv2.findContours(threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # find the index of the biggest contour
    max_area = 0
    num_contours = len(contours)
    for i in range(num_contours):
        
        curr_contour = contours[i]
            
        area = cv2.contourArea(curr_contour)
        
        if(area > max_area):
            max_area = area
            largest_contour_index = i

    largest_contour = contours[largest_contour_index]
    # hull = set of convex points on a curve
    hull = cv2.convexHull(largest_contour)
    moments = cv2.moments(largest_contour)

    if moments['m00'] != 0:
        cx = int(moments['m10']/moments['m00']) # cx = M10/M00
        cy = int(moments['m01']/moments['m00']) # cy = M01/M00

    center = (cx,cy) # center of contour
    cv2.circle(frame, center, 5, [0,0,255], 2) # draw a red open circle at center of contour
    drawing = np.zeros(frame.shape, np.uint8)
    cv2.drawContours(drawing,[largest_contour], 0, (255,0,0), 2)
    cv2.drawContours(drawing,[hull], 0, (0,255,0), 2)

    cv2.imshow('Contours', drawing)
    
    # find convexity defects (the points that are not convex, returnPoints must be false for this)
    hull_2 = cv2.convexHull(largest_contour, returnPoints = False)
    
    defects = cv2.convexityDefects(largest_contour, hull_2)

    num_defects = defects.shape[0]
    furthest_defects = []
    for i in range(num_defects):
        
        s,e,f,d = defects[i,0]
        
        start = tuple(largest_contour[s][0])
        end = tuple(largest_contour[e][0])
        far = tuple(largest_contour[f][0])
        
        furthest_defects.append(far)

    defect_distances_from_center = []
    for i in range(len(furthest_defects)):
        x =  np.array(furthest_defects[i])
        distance = get_distance_between_points(center, x)
        defect_distances_from_center.append(distance)
        cv2.circle(frame, furthest_defects[i], 5, [255,0,0], -1)

    # finds the average of the 3 closest distances from the center
    sorted_defect_distances = sorted(defect_distances_from_center)
    average_defect_distance = np.mean(sorted_defect_distances[0:2])
    
    # finds fingertip points from the hull. Since there will be multiple hull points on the finger tips, this piece of code considers all points within 30 pixels of each other as one point
    finger = []
    for i in range(len(hull)-1):
        if (np.absolute(hull[i][0][0] - hull[i+1][0][0]) > 80) or ( np.absolute(hull[i][0][1] - hull[i+1][0][1]) > 40):
            if hull[i][0][1] <500 :
                finger.append(hull[i][0])

    
    #The fingertip points are 5 hull points with largest y coordinates  
    fingers =  sorted(finger,key=lambda x: x[1])   
    fingers = finger[0:5]

    #Calculate distance of each finger tip to the center mass
    fingertip_distances = []
    
    for i in range(len(fingers)):
        distance = get_distance_between_points(fingers[i], center)
        fingertip_distances.append(distance)

    #This finds the number of raised fingers (if fingertip points are greater than the average defect distance by 120 pixels, then finger is raised
    num_raised_fingers = 0
    for i in range(len(fingers)):
        if fingertip_distances[i] > average_defect_distance + 120:
            num_raised_fingers = num_raised_fingers + 1

    # radius for cicles to be drawn)
    radius = circleRad.get() # also line width
    
    # finds the coordinates of the furthest fingertip
    curr_farthest_point_from_center = find_farthest_point_from_contour_center(largest_contour, defects, center)

    # another way to find the coordinates of the furthest fingertip is commented out below 
    #index_max_fingertip_dist = fingertip_distances.index(max(fingertip_distances))
    #curr_farthest_point_from_center = tuple(fingers[index_max_fingertip_dist])


    # draws a circle at the current finger position on the canvas (like a cursor following the finger)
    if(curr_farthest_point_from_center is not None):
        if(first_run):
            first_run = False
            finger_cursor = canv.create_oval(curr_farthest_point_from_center[0], curr_farthest_point_from_center[1], curr_farthest_point_from_center[0]+radius, curr_farthest_point_from_center[1]+radius, outline = 'purple')
        else:
            if(prev_farthest_point_from_center is not None):
                dx = curr_farthest_point_from_center[0] - prev_farthest_point_from_center[0]
                dy = curr_farthest_point_from_center[1] - prev_farthest_point_from_center[1]
                canv.move(finger_cursor, dx, dy)
                
    # dont draw if inside rectangle to change color
    inside_rect = False
    
    if (curr_farthest_point_from_center is not None and prev_farthest_point_from_center is not None):
        # only if the previously drawn circle doesn't overlap with the new one draw the point on the canvas)
        #if(not check_circle_overlap(prev_farthest_point_from_center, curr_farthest_point_from_center, radius)):
            cv2.circle(frame, curr_farthest_point_from_center, 5, [255,0,255], -1)
            if(check_point_inside_rectangle(curr_farthest_point_from_center, blue_rect_xy, blue_rect_x2y2)):
                color = '#000000fff' #pure blue
                inside_rect = True
            elif(check_point_inside_rectangle(curr_farthest_point_from_center, red_rect_xy, red_rect_x2y2)):
                color = '#fff000000' #pure red
                inside_rect = True
            elif(check_point_inside_rectangle(curr_farthest_point_from_center, green_rect_xy, green_rect_x2y2)):
                color = '#000fff000' #pure green
                inside_rect = True
                
            # draw on canvas only if one finger is raised
            print(num_raised_fingers)
            if (num_raised_fingers < 3 and not inside_rect):
                canv.create_line(prev_farthest_point_from_center[0], prev_farthest_point_from_center[1], curr_farthest_point_from_center[0], curr_farthest_point_from_center[1], fill = color, width = radius)
    
    cv2.imshow('Live Video', frame)

    # clear the canvas to draw again (gesture for this is 4 fingers)
    if (num_raised_fingers == 4):
        canv.delete(tk.ALL)
        draw_rects()
        first_run = True
    
    prev_farthest_point_from_center = curr_farthest_point_from_center
    
cap.release()
cv2.destroyAllWindows()
root.destroy()
