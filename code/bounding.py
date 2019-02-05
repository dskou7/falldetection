import cv2
import os
import numpy as np

## Get video from data and store in a cv2 videoCapture object
path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(path, "../data/fall1.mp4")
reel = cv2.VideoCapture(path)
## Make background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()
## While loop for processing each frame until no more frames in video
while(1):
  _, frame = reel.read()
  # If frame is of none type then you've reached the end of the video
  if(type(frame) == type(None)): 
    break
  # extract the foreground using the background subtractor.
  fgmask = fgbg.apply(frame) 

  # TODO - Insert Kalman filter (feed fgmask into it). Use it's output for next thing
  # TODO - Insert optical flow and possibly occlusion detection
  # TODO - Insert other pipeline pieces here.

  # discover all the contours in the image. This makes a big list of contours
  contours, _ = cv2.findContours(
      fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  ## Draw a bounding rectangle around what we think the person is. 
  ## Currently, I'm just taking the largest contour and assuming that's the person. Could be better
  contours = sorted(contours, key=cv2.contourArea, reverse=True)
  x, y, w, h = cv2.boundingRect(contours[0])
  cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
  ## Show the current frame, with any additional things we've drawn superimposed onto the image
  cv2.imshow('result', frame)
  cv2.waitKey(0) # This just pauses until you press a key. I'm not sure which keys work, but i know 'n' does

reel.release()
cv2.destroyAllWindows()
