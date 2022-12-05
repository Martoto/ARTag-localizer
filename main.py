#!/usr/bin/env python3

# Import standard libraries
import cv2
import numpy as np
import math
import json
import socket
import statistics

from copy import deepcopy



 # Define constants for entire project
ref_dist = 10 #in centimeters
ref_dimension = 400
t_y = 80
p_y = 1300
h_y = 1500
p = 0.147
grid = (10,9)
alturarobo = 19
alturacamera = 240


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
        if 3750 < contour_area < 10000 and len(contour_poly_curve) == 4:
                x, y, w, h = cv2.boundingRect(contour_poly_curve)
                x = x + w/2
                y = y + h/2
                true_y = y + (t_y*((p_y - y)/h_y))
                ratio= float(w)/h
                #print(x, y)
                if ratio>=0.9 and ratio<=1.1 :
                    # Warp the video frame
                    #print(contour_poly_curve)
                    if (contour_poly_curve[3][0][1] < contour_poly_curve[1][0][1]):	
                        aux_curve = deepcopy(contour_poly_curve)
                        contour_poly_curve[0], contour_poly_curve[1], contour_poly_curve[2], contour_poly_curve[3] = aux_curve[3],  aux_curve[0], aux_curve[1], aux_curve[2]


                    
                    mat, _ = si.get_h_matrices(contour_poly_curve, ref_dimension, ref_dimension)
                    vf_warp = cv2.transpose(cv2.warpPerspective(video_frame, mat, (ref_dimension, ref_dimension)))
                    #espelha o arTag. 
                    _,vf_warp = cv2.threshold(cv2.cvtColor(vf_warp, cv2.COLOR_BGR2GRAY), 150, 255, 0)
                    #cv2.imwrite("sampleWarp.png",vf_warp)
                    # Get orientation and tag ID
                    

                    orientation = detector.get_tag_orientation(vf_warp)

                    p1, p2 = contour_poly_curve[orientation-1][0], contour_poly_curve[orientation][0]
                    vec = (p2[0] - p1[0], p2[1] - p1[1])
                    ang = detector.angle_of_vectors(vec,(1,0))

                    #print(orientation)
                    #print(p, rows, p*rows)

                    dx = x-cols/2
                    correction = (1-alturarobo/alturacamera)
                    true_x = cols/2+dx*correction
                    print(dx, correction, true_x, cols/2)
                    # Draw the selected Contour matching the criteria fixed

                    video_frame = cv2.drawContours(video_frame, [contour], 0, (0, 0, 225), 1)
                    video_frame = cv2.drawMarker(video_frame,((int)(x),(int)(y)),(0, 0, 225))
                    #cv2.drawMarker(video_frame,(contour_poly_curve[0][0]),(0, 255, 0))
                    #cv2.drawMarker(video_frame,(contour_poly_curve[1][0]),(255, 0, 0))
                    #cv2.drawMarker(video_frame,(p2),(0, 255, 0))
                    video_frame = cv2.line(video_frame, ((int)(x),(int)(y)), ((int)(x),(int)(true_y)),(255,0,0), thickness=2)
                    video_frame = cv2.line(video_frame, p1, p2,(0,255,0), thickness=3)
                    video_frame = cv2.line(video_frame, p1, (p1[0] + (int)(math.dist(p1,p2)),p1[1]),(0,255,0),thickness=3)
                    cv2.putText(video_frame, f"({p*true_x:.1f}, {p*true_y:.1f})", (contour_poly_curve[0][0][0] - 50,
                                                               contour_poly_curve[0][0][1] + 50),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 225), 2, cv2.LINE_AA)
                   
                    #tag_id = detector.get_tag_id(vf_warp_gray, orientation)
                    # Display tag ID on each frame

                    #print(detector.return_grid((9,10),(x,y),(cols,rows)))
                    return video_frame, (true_x,true_y), ang, orientation
    return video_frame, (0,0), 0, 0
                    

def dang(a, b):
    k = (a-b) %360 
    return min(360-k, k)

if __name__ == '__main__':
    import server 

   # Import custom function scripts
    from utils import detector, superimpose as si
    # Create cv objects for video and Mask image

    serverinstance = server.run_server()

    tag = cv2.VideoCapture(0)
    tag.set(cv2.CAP_PROP_FRAME_WIDTH , 2304)
    tag.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)
    #video_frame = cv2.imread("./sample.png")
    h_mat = np.load("/home/arena/Documents/GitHub/Robinho/Robinho_Webapp/RobinhoImageProcessing/h_mat.npy")

    ACCLEN = 3
    acc = [None] * ACCLEN

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
        if (ang > 0):
            ang = 360 - ang
        else:
            ang *= -1



        pose = (p*x, p*y, ang)
        print(pose)

        if (math.isnan(ang)) or ((not x) and (not y) and (not ang)):
            pose = None

        acc.append(pose)

        if acc[-1] and acc[-2] and acc[-3]:
            x_stdev = statistics.stdev([acc[-1][0],acc[-2][0],acc[-3][0],])
            y_stdev = statistics.stdev([acc[-1][1],acc[-2][1],acc[-3][1],])
            # https://en.wikipedia.org/wiki/Circular_mean
            ang_x = math.sin(math.radians((acc[-1][2]+acc[-2][2]+acc[-3][2])/3))
            ang_y = math.cos(math.radians((acc[-1][2]+acc[-2][2]+acc[-3][2])/3))
            ang_mean = math.degrees(math.atan2(ang_x, ang_y))
            #print("a:", ang_x, ang_y, ang_mean)
            max_ang_d = max(dang(acc[-1][2], ang_mean), dang(acc[-2][2], ang_mean), dang(acc[-3][2], ang_mean))
            #print("math: ", x_stdev, y_stdev, ang_mean, max_ang_d)
            
            if x_stdev < 2.0 and y_stdev < 2.0 and max_ang_d < 2.0:
                x = statistics.mean([acc[-1][0],acc[-2][0],acc[-3][0],])
                y = statistics.mean([acc[-1][1],acc[-2][1],acc[-3][1],])
                ang = ang_mean
                out_x = round(x)
                out_y = round(y)
                out_z = round((256*((ang+360)%360)/360.0)%256)

                out = [out_x, out_y, out_z]
                print("sending: ", bytes([0x0] + out).hex())
                server.send_pose(out)
            else:
                print('WARNING: Outlier value detected')
        else:
            print('ERROR: No TAG detected')

        acc = acc[-3:]
        
        imghsv = cv2.cvtColor(vf_original, cv2.COLOR_BGR2HSV).astype("float32")
        (h, s, v) = cv2.split(imghsv)
        s = s*3
        s = np.clip(s,0,255)
        imghsv = cv2.merge([h,s,v])
        imgbgr = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2BGR)


        cv2.imwrite('/home/arena/Documents/GitHub/Robinho/Robinho_Webapp/images/feed.png', imgbgr)
        #cv2.imwrite('./feed.png', vf_original)

    #ESP_server.close()  # close the connection

