# program to capture single image from webcam in python
  
# importing OpenCV library
import cv2 as cv
import urllib.request
  
# initialize the camera
# If you have multiple camera connected with 
# current device, assign a value in cam_port 
# variable according to that

cam_port = "http://192.168.100.253:8081/"
cam = cv.VideoCapture(2)
  
# reading the input using the camera
result, image = cam.read()
  
# If image will detected without any error, 
# show result
if result:
  
    # saving image in local storage
    cv.imwrite("GeeksForGeeks.png", image)
  
    # If keyboard interrupt occurs, destroy image 
    # window
    cv.waitKey(0)
    cv.destroyWindow("GeeksForGeeks")
  
# If captured image is corrupted, moving to else part
else:
    print("No image detected. Please! try again")