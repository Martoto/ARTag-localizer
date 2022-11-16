import cv2 as cv
import numpy as np
import math

def find_contours(img_frame):
    """
    find and draw contours on the main frame
    :param img_frame: a frame from the video
    :return: image with contours and the contours matrix
    """
    img = cv.cvtColor(img_frame, cv.COLOR_BGR2GRAY)
    frame_thresh = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
    contours, _ = cv.findContours(frame_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours

def draw_grid(grid_shape, scale, color=(255, 255, 255), thickness=2):
    img = np.zeros((scale[1],scale[0], 1), dtype=np.uint8)
    h, w, _ = img.shape
    cols, rows = grid_shape
    dy, dx = h / rows, w / cols

    # draw vertical lines
    for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
        x = int(round(x))
        cv.line(img, (x, 0), (x, h), color=color, thickness=thickness)

    # draw horizontal lines
    for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
        y = int(round(y))
        cv.line(img, (0, y), (w, y), color=color, thickness=thickness)

    return img

def return_grid(grid = (2, 2), pos = (0, 0), res = (640, 640)):
    """ get grid in which (pos_x, pos_y) is in
    :param (cols, rows) number of grid cells (pos_x,pos_y) position to identify. (height, width) = resolution
    :return: matrix identifying the position of the point
    """
    
    w , h = res[1],res[0]
    cols, rows = grid[0],grid[1]
    x, y = pos[0],pos[1]

    cell_h, cell_w = h/rows, w/cols

    #grid_array = np.zeros((cols,rows), dtype=int)
    x, y = (int)(x/cell_w), (int)(y/cell_h)
    #grid_array[x, y] = 1
    return (x,y)

def distFromCell(pos, res, cell, grid):
    w , h = res[1],res[0]
    cols, rows = grid[0],grid[1]

    cell_h, cell_w = h/rows, w/cols
    cell_pos = (cell_w*cell[0], cell_h*cell[1])

    return (cell_pos[0] - pos[0],cell_pos[1] - pos[1])

    
    
def angle_of_vectors(vec1,vec2):
    a,b = vec1
    c,d = vec2
    dotProduct = a*c + b*d
        # for three dimensional simply add dotProduct = a*c + b*d  + e*f 
    modOfVector1 = math.sqrt( a*a + b*b)*math.sqrt(c*c + d*d) 
        # for three dimensional simply add modOfVector = math.sqrt( a*a + b*b + e*e)*math.sqrt(c*c + d*d +f*f) 
    angle = dotProduct/modOfVector1
    angleInDegree = math.degrees(math.acos(angle))
    return angleInDegree
     


def get_tag_orientation(img_frame):
    """ get orientation from the image frame
    :param img_frame: image frame from the video
    :return: orientation of the tag
    """
    # Check get_H_matrix function in superimpose for orientation notation
    orientations = {0: 0, 1: 0, 2: 0, 3: 0}
    # Orientation: Bottom Right
    for i in range(250, 301):
        for j in range(250, 301):
            orientations[0] += img_frame[i, j]
    # Orientation: Bottom Left
    for i in range(250, 301):
        for j in range(100, 151):
            orientations[1] += img_frame[i, j]
    # Orientation: Top Right
    for i in range(100, 151):
        for j in range(250, 301):
            orientations[2] += img_frame[i, j]
    # Orientation: Top Left
    for i in range(100, 151):
        for j in range(100, 151):
            orientations[3] += img_frame[i, j]

    return max(orientations, key=orientations.get)


def get_tag_id(img_frame, orientation):
    """
    :param img_frame: current frame of the video
    :param orientation: orientation of the tag
    :return: tag ID
    """
    tag_id = ''
    keys = []
    # Check get_H_matrix function in superimpose.py for orientation notation
    if orientation == 0:
        keys = [1, 0, 2, 3]
    elif orientation == 1:
        keys = [3, 1, 0, 2]
    elif orientation == 2:
        keys = [2, 3, 1, 0]
    elif orientation == 3:
        keys = [0, 2, 3, 1]
    structure = {0: [200, 250, 200, 250], 1: [150, 200, 200, 250], 2: [200, 250, 150, 200], 3: [150, 200, 150, 200]}

    total = 0
    for key in keys:
        for i in range(structure[key][0], structure[key][1]):
            for j in range(structure[key][2], structure[key][3]):
                total += img_frame[i][j]

        if (total / 2500) > 220:
            tag_id += '1'
        else:
            tag_id += '0'
    return tag_id
