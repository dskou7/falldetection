import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

## Get video from data and store in a cv2 videoCapture object
path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(path, "../data/fall1.mp4")
reel = cv2.VideoCapture(path)
## Make background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()
## Initialize variables
n_frames = length = int(reel.get(cv2.CAP_PROP_FRAME_COUNT))
rect_ratios = np.zeros(n_frames)

## While loop for processing each frame until no more frames in video
curr_frame = 0
while(curr_frame < n_frames):
  _, frame = reel.read()
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
  rect_ratios[curr_frame] = w / h
  cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
  ## Show the current frame, with any additional things we've drawn superimposed onto the image
  cv2.imshow('result', frame)
  if(curr_frame > 13):
    cv2.waitKey(0) # This just pauses until you press a key. I'm not sure which keys work, but i know 'n' does
  curr_frame += 1

reel.release()
cv2.destroyAllWindows()

## Analyze some data
plt.scatter(range(n_frames), rect_ratios)
plt.xlabel('Video Frame')
plt.ylabel('Rectangle Ratio')
plt.show()
'''
Conclusions: Looks like rectangle ratio is indeed a good variable for this. In fall1.mp4, the subjects fall begins
around frame 15 or so. That's also the point where we begin to see a sharp upward trend in rect_ratio. The subject
is pretty tall and lanky though, which makes him an ideal subject for this type of detection. Wider, shorter subjects
will see less difference in rect_ratio between standing up and laying down.
'''
