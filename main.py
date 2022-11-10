# Import standard libraries
import cv2
import numpy as np
from sys import argv
from copy import deepcopy
# Import custom function scripts
from utils import draw_3d, detector, superimpose as si




if __name__ == '__main__':
    # Define constants for entire project
    ref_dist = 10 #in centimeters
    ref_dimension = 400
    grid = (9,10)
    # Create cv objects for video and Mask image
    """
    tag = cv2.VideoCapture(0)
    tag.set(cv2.CAP_PROP_FRAME_WIDTH , 2304)
    tag.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)
    """
    mask = cv2.imread("./ArenaMask.png")
    video_frame = cv2.imread("./sample.png")
    h_mat = np.load("./h_mat.npy")
    # Define output video object
    total_frames = 0
    # Begin loop for iterate through each frame of the video
    # Store original frame for comparison

    rows, cols, channels = video_frame.shape
    newcol = (int)(rows*10/9)
    video_frame = cv2.transpose(cv2.warpPerspective(video_frame, h_mat, (cols, cols)))[0:rows, 0:newcol]
    
    cv2.imwrite("sampleWarped.png",video_frame)
    vf_original = deepcopy(video_frame)
    #vf_grayscale = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    # Store all size parameters of the frame

    # Get contours from the video frame

    contours = detector.find_contours(video_frame)
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        contour_poly_curve = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, closed=True), closed=True)
        if 1000 < contour_area < 10000 and len(contour_poly_curve) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                x = x + w/2
                y = y + h/2
                ratio= float(w)/h
                #print(x, y)
                if ratio>=0.9 and ratio<=1.1 :
                    p = ref_dist/h
                    print(p, rows, p*rows)
                    # Draw the selected Contour matching the criteria fixed

                    cv2.drawContours(vf_original, [contour], 0, (0, 0, 225), 1)
                    # Warp the video frame

                    #vf_warp = cv2.warpPerspective(video_frame, h_mat, (ref_dimension, ref_dimension))
                    #vf_warp_gray = cv2.cvtColor(vf_warp, cv2.COLOR_BGR2GRAY)
                    # Get orientation and tag ID

                    #orientation = detector.get_tag_orientation(vf_warp_gray)
                    #tag_id = detector.get_tag_id(vf_warp_gray, orientation)
                    # Display tag ID on each frame

                    #print(detector.return_grid((9,10),(x,y),(cols,rows)))
    """
    while True:
        # Read the video frame by frame
        video_frame_exists, video_frame = tag.read()
        # Exit when the video file ends
        if not video_frame_exists:
            break
        total_frames += 1
        
        cv2.imshow('feed',vf_original)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Destroy all cv objects
    tag.release()
    cv2.destroyAllWindows()
    """




def processFrame(video_frame):
    # Store original frame for comparison

    rows, cols, channels = video_frame.shape
    video_frame = cv2.warpPerspective(video_frame, h_mat, (cols, cols))
    vf_original = deepcopy(video_frame)
    #vf_grayscale = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    # Store all size parameters of the frame

    # Get contours from the video frame

    contours = detector.find_contours(video_frame)
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        contour_poly_curve = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, closed=True), closed=True)
        if 1000 < contour_area < 10000 and len(contour_poly_curve) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                x = x + w/2
                y = y + h/2
                ratio= float(w)/h
                print(x, y)
                if ratio>=0.9 and ratio<=1.1 :
                    p = ref_dist/h
                    print(p*rows)
                    # Draw the selected Contour matching the criteria fixed

                    cv2.drawContours(vf_original, [contour], 0, (0, 0, 225), 1)
                    # Warp the video frame

                    #vf_warp = cv2.warpPerspective(video_frame, h_mat, (ref_dimension, ref_dimension))
                    #vf_warp_gray = cv2.cvtColor(vf_warp, cv2.COLOR_BGR2GRAY)
                    # Get orientation and tag ID

                    #orientation = detector.get_tag_orientation(vf_warp_gray)
                    #tag_id = detector.get_tag_id(vf_warp_gray, orientation)
                    # Display tag ID on each frame

                    print(detector.return_grid((9,10),(x,y),(cols,rows)))
                    
                    
