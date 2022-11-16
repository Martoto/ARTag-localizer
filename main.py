# Import standard libraries
import cv2
import numpy as np
import math
import json
import socket

from copy import deepcopy

ESP_server = socket.socket()
ESP_server.bind(("0.0.0.0", 5072))
ESP_server.listen(1)



 # Define constants for entire project
ref_dist = 10 #in centimeters
ref_dimension = 400
p = 0.147
grid = (10,9)


pos = {
    "pos": [0, 0],
    "cell": [ 0,0],
    "error": [0, 0],
    "degrees": 0,
    "heading": 0,
    "ang_err": 0
  }



def warpFrame(video_frame):
    rows, cols, _ = video_frame.shape
    newcol = (int)(rows*10/9)
    video_frame = cv2.transpose(cv2.warpPerspective(video_frame, h_mat, (cols, cols)))[0:rows, 0:newcol]
    return video_frame
    #cv2.imwrite("sampleWarped.png",video_frame)

def resizeFrame(video_frame, scale_percent = 50):
    width = int(video_frame.shape[1] * scale_percent / 100)
    height = int(video_frame.shape[0] * scale_percent / 100)
    dim = (width, height)
  
    # resize image
    return cv2.resize(video_frame, dim, interpolation = cv2.INTER_AREA)

def processFrame(video_frame):
    #vf_grayscale = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    # Store all size parameters of the frame

    # Get contours from the video frame
    rows, cols, _ = video_frame.shape

    contours = detector.find_contours(video_frame)
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        contour_poly_curve = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, closed=True), closed=True)
        if 1000 < contour_area < 8000 and len(contour_poly_curve) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                x = x + w/2
                y = y + h/2
                ratio= float(w)/h
                #print(x, y)
                if ratio>=0.9 and ratio<=1.1 :
                    # Warp the video frame
                    #print(contour_poly_curve)
                    p1, p2 = contour_poly_curve[2][0], contour_poly_curve[3][0]
                    vec = (p1[0] - p2[0], p1[1] - p2[1])
                    ang = detector.angle_of_vectors(vec,(1,0))
                    mat, _ = si.get_h_matrices(contour_poly_curve, ref_dimension, ref_dimension)
                    vf_warp = cv2.warpPerspective(video_frame, mat, (ref_dimension, ref_dimension))
                    _,vf_warp = cv2.threshold(cv2.cvtColor(vf_warp, cv2.COLOR_BGR2GRAY), 150, 255, 0)
                    #cv2.imwrite("sampleWarp.png",vf_warp)
                    # Get orientation and tag ID

                    orientation = detector.get_tag_orientation(vf_warp)
                    #print(orientation)
                    #print(p, rows, p*rows)
                    # Draw the selected Contour matching the criteria fixed

                    cv2.drawContours(video_frame, [contour], 0, (0, 0, 225), 1)
                    cv2.drawMarker(video_frame,((int)(x),(int)(y)),(0, 0, 225))
                    cv2.drawMarker(video_frame,(p1),(0, 255, 0))
                    cv2.drawMarker(video_frame,(p2),(0, 255, 0))
                    video_frame = cv2.line(video_frame, p1, p2,(0,255,0), thickness=3)
                    video_frame = cv2.line(video_frame, p1, (p1[0] + (int)(math.dist(p1,p2)),p1[1]),(0,255,0),thickness=3)
                    cv2.putText(video_frame, "( i, j ) = " + str(detector.return_grid((10,9),(x,y),(rows,cols))), (contour_poly_curve[0][0][0] - 50,
                                                               contour_poly_curve[0][0][1] - 50),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 225), 2, cv2.LINE_AA)
                   
                    #tag_id = detector.get_tag_id(vf_warp_gray, orientation)
                    # Display tag ID on each frame

                    #print(detector.return_grid((9,10),(x,y),(cols,rows)))
                    return video_frame, (x,y), ang, orientation
    return video_frame, (0,0), 0, 0
                    


if __name__ == '__main__':
   # Import custom function scripts
    from utils import detector, superimpose as si
    # Create cv objects for video and Mask image


    PC_link, addr = ESP_server.accept()
    PC_link.setblocking(0)
    #print('client connected from', addr)

    tag = cv2.VideoCapture(0)
    tag.set(cv2.CAP_PROP_FRAME_WIDTH , 2304)
    tag.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)
    mask = cv2.imread("./ArenaMask.png")
    video_frame = cv2.imread("./sample.png")
    h_mat = np.load("./h_mat.npy")

    while True:
        # Read the video frame by frame
        video_frame_exists, video_frame = tag.read()
        # Exit when the video file ends
        if not video_frame_exists:
            break
        
        vf_original = deepcopy(video_frame)
        vf_original = warpFrame(vf_original)
        vf_original, (x,y), ang, orientation = processFrame(vf_original)
        rows, cols, _ = vf_original.shape 
        mask = cv2.bitwise_not(detector.draw_grid(grid, (cols,rows)))   
        vf_original = cv2.bitwise_and(vf_original,vf_original,mask=mask)

        #cell and error = distance from cell center
        cell = detector.return_grid(grid,(x,y),(cols,rows))
        error = detector.distFromCell((x,y),(cols,rows),cell,grid)
        vf_original = resizeFrame(vf_original)

        #print((p*x,p*y), p*error[0], p*error[1])

        """
        pos.pos = (p*x,p*y)
        pos.erros = (p*error[0], p*error[1])
        pos.cell = cell
        pos.degrees = ang
        pos.heading = orientation
        pos.ang_error = 90*orientation - ang
        """

        out_x = round(p*x)
        out_y = round(p*y)
        out_z = round(ang)

        data = str((out_x, out_y, out_z))
        print(orientation)
        #print(data)

        cv2.imwrite('/home/arena/Documents/GitHub/Robinho/Robinho_Webapp/images/feed.png', vf_original)

    #ESP_server.close()  # close the connection


                    
