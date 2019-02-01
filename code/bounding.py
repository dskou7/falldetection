import cv2
import numpy as np



# while(1):

#     cv2.imshow('fgmask', frame)
#     cv2.imshow('frame', fgmask)

#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break

fgbg = cv2.createBackgroundSubtractorMOG()
reel = cv2.VideoCapture("data/fall1.mp4")
while(1):
  _, frame = reel.read()
  if(type(frame) == type(None)):
    break
#   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#   graycp = np.copy(gray)
  fgmask = fgbg.apply(frame)
  contours, _ = cv2.findContours(
      fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contours = sorted(contours, key=cv2.contourArea, reverse=True)
#   cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
  x, y, w, h = cv2.boundingRect(contours[0])
  cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
  cv2.imshow('result', frame)
  cv2.waitKey(0)

reel.release()
cv2.destroyAllWindows()
