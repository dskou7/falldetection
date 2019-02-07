import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

## Get video from data and store in a cv2 videoCapture object
path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(path, "../data/fall5.mp4")
reel = cv2.VideoCapture(path)
## Make background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()
## Initialize variables
n_frames = length = int(reel.get(cv2.CAP_PROP_FRAME_COUNT))
rect_ratios = np.zeros(n_frames)
ellipse_angles = np.zeros(n_frames)

## While loop for processing each frame until no more frames in video
curr_frame_i = 0
while(curr_frame_i < n_frames):
  _, frame = reel.read()
  # extract the foreground using the background subtractor.
  fgmask = fgbg.apply(frame)

  # TODO - Insert Kalman filter (feed fgmask into it). Use it's output for next thing
  # TODO - Insert optical flow and possibly occlusion detection
  # TODO - Insert other pipeline pieces here.

  # discover all the contours in the image. This makes a big list of contours
  contours, _ = cv2.findContours(
      fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  ## Draw a bounding rectangle and ellipse around what we think the person is. Store angle and ratio variables for later
  ## Currently, I'm just taking the largest contour and assuming that's the person. Could be better
  contours = sorted(contours, key=cv2.contourArea, reverse=True)
  x, y, w, h = cv2.boundingRect(contours[0])
  cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
  # fitEllipse throws an error if the contour doesn't have at least 5 points. This occurs on first frame often (just noise)
  if(len(contours[0]) >= 5):
    ellipse = cv2.fitEllipse(contours[0])
    cv2.ellipse(frame, ellipse, (0, 0, 255), 2)
    ellipse_angles[curr_frame_i] = ellipse[2]
  rect_ratios[curr_frame_i] = w / h
  print('curr_frame_i:', curr_frame_i)
  print('rect_ratio: {0} | ellipse_angle: {1}'.format(
      rect_ratios[curr_frame_i], ellipse_angles[curr_frame_i]))
  ## Show the current frame, with any additional things we've drawn superimposed onto the image
  cv2.imshow('result', frame)
  # if(curr_frame_i > 10):
  # This just pauses until you press a key. I'm not sure which keys work, but i know 'n' does
  cv2.waitKey(0)
  curr_frame_i += 1

reel.release()
cv2.destroyAllWindows()


# TODO - Fill missing values (0s) for ellipse angles. Just use a linear interpolation

#### Analyze some data
## Rectangle ratio
plt.scatter(range(n_frames), rect_ratios)
plt.title('Rectangle Ratio Over Time')
plt.xlabel('Video Frame')
plt.ylabel('Rectangle Ratio')
plt.show()
'''
Observations: Looks like rectangle ratio is indeed a good variable for this. 
--- fall1.mp4 ---
the subjects fall begins around frame 15 or so. That's also the point where we begin to see a sharp upward trend in rect_ratio. The subject
is pretty tall and lanky though, which makes him an ideal subject for this type of detection. Wider, shorter subjects
will see less difference in rect_ratio between standing up and lying down.
--- fall2.mp4 ---
When the subject first enters the scene, we pick up her hand and foot as the entire subject. This will cause some weird data points when people enter the view of the camera, so we may want to come up with a way to handle this. We may not need to though with the rest of the pipeline, we'll see
--- fall5.mp4 ---
A bit of noise, but not as noisy as the ellipse. Kalman filter should help
'''
## Ellipse angle
plt.scatter(range(n_frames), ellipse_angles)
plt.title('Ellipse Angle Over Time')
plt.xlabel('Video Frame')
plt.ylabel('Ellipse Angle w/ Y-axis')
plt.show()
'''
Observations: Of course, the angle of the ellipse represents if the person is lying down or not. Values close to 90 
represent lying down (perpendicular to y-axis). The derivative (the third variable we need) is critical to be able 
to detect a rapid change from standing to lying down
--- fall2.mp4 ---
Similar observation to the rectangle ratio for this video.
--- fall5.mp4 ---
This was super noisy for some reason! This is a good video to test the effect of the kalman filter, because it should help a lot with this noise
'''
