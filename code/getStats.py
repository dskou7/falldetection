import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

############## Function declarations ##############
def processVideo(full_path):
  reel = cv2.VideoCapture(full_path)
  ## Make background subtractor
  fgbg = cv2.createBackgroundSubtractorMOG2()
  ## Initialize variables
  n_frames = length = int(reel.get(cv2.CAP_PROP_FRAME_COUNT))
  rect_w = np.zeros(n_frames)
  rect_h = np.zeros(n_frames)
  ellipse_angles = np.zeros(n_frames)

  curr_frame_i = 0
  while(curr_frame_i < n_frames):
    _, frame = reel.read()
    fgmask = fgbg.apply(frame)

    ## TODO - replace this with MXNET human detection
    contours, _ = cv2.findContours(
        fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(contours[0])
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # fitEllipse throws an error if the contour doesn't have at least 5 points. This occurs on first frame often (just noise)
    if(len(contours[0]) >= 5):
      ellipse = cv2.fitEllipse(contours[0])
      cv2.ellipse(frame, ellipse, (0, 0, 255), 2)
      ellipse_angles[curr_frame_i] = ellipse[2]
    rect_w[curr_frame_i] = w
    rect_h[curr_frame_i] = h
    # print('curr_frame_i:', curr_frame_i)
    # print('rect_ratio: {0} | ellipse_angle: {1} '.format(rect_ratios[curr_frame_i], ellipse_angles[curr_frame_i]))
    curr_frame_i += 1

  reel.release()
  cv2.destroyAllWindows()
  return n_frames, rect_h, rect_w, ellipse_angles

def getStatsForVideo(n_frames, rect_h, rect_w, ellipse_angles, video_name):
  def removeOutliers(nparray, outlierConstant=1.5):
      upper_quartile = np.percentile(nparray, 75)
      lower_quartile = np.percentile(nparray, 25)
      IQR = (upper_quartile - lower_quartile) * outlierConstant
      quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
      result = nparray[np.where(
          (nparray >= quartileSet[0]) & (nparray <= quartileSet[1]))]
      return result

  '''
  Calculates the biggest change that occurs in a variable over time, after removing outliers. 
  @param nparray: an np array representing the variable over time
  @param outlierConstant: The factor for how far out from the IQR a value must be to be considered an outlier
  @return: the change in the variable. Positive means it increased with time, negative it decreased with time
  '''
  # If the min is at an earlier frame than the max, the variable has increased so direction should be positive.
  # RISK - but what if there are multiple mins/maxes throughout the video? argmin/argmax only returns the first occurence
  # RISK 2 (CRITICAL) - the biggest change may be an uninteresting one (IE the biggest change in h for a given clip may be negative, which we don't care about) but that doesn't mean the second biggest change wouldn't be a significant interesting one. This is a problem
  def calcBiggestChange(nparray, outlierConstant=1.5):
    clean_a = removeOutliers(nparray, outlierConstant)
    min_i = np.argmin(clean_a)
    max_i = np.argmax(clean_a)
    direction = 1 if min_i < max_i else -1
    delta = direction * (clean_a[max_i] - clean_a[min_i])
    return delta

  delta_h = calcBiggestChange(rect_h)
  delta_w = calcBiggestChange(rect_w)
  rect_ratios = rect_w / rect_h
  delta_ratio = calcBiggestChange(rect_ratios)
  delta_angle = calcBiggestChange(ellipse_angles) # may want to reduce the outlier constant for angle. outlierConstant=.75 worked well on this one clip.

  # Title will just be y_label concatenated with ' Over Time'
  # def plotVariableVsFrame(variable, n_frames, y_label):
  #   plt.scatter(range(n_frames), variable)
  #   plt.title(y_label + ' Over Time')
  #   plt.xlabel('Video Frame')
  #   plt.ylabel(y_label)
  #   plt.show()

  ## Make sure outlier removal is working as intended
  # plotVariableVsFrame(rect_h, n_frames, 'Rectangle Height')
  # clean_h = removeOutliers(rect_h)
  # plotVariableVsFrame(clean_h, len(clean_h), 'Cleansed height')
  # plotVariableVsFrame(rect_w, n_frames, 'Rectangle Width')
  # clean_w = removeOutliers(rect_w)
  # plotVariableVsFrame(clean_w, len(clean_w), 'Cleansed width')
  # plotVariableVsFrame(rect_ratios, n_frames, 'Rectangle Ratio')
  # clean_ratio = removeOutliers(rect_ratios)
  # plotVariableVsFrame(clean_ratio, len(clean_ratio), 'Cleansed ratio')
  # plotVariableVsFrame(ellipse_angles, n_frames, 'Ellipse Angle')
  # clean_angle = removeOutliers(ellipse_angles) 
  # plotVariableVsFrame(clean_angle, len(clean_angle), 'Cleansed angle')

  data = {'Video': [video_name], 'Label': [1], 'Delta h': [delta_h], 'Delta w': [delta_w], 'Delta ratio': [delta_ratio], 'Delta angle': [delta_angle] }
  df = pd.DataFrame(data, columns=['Video', 'Label', 'Delta h', 'Delta w', 'Delta ratio', 'Delta angle'])
  return df

## Fall videos
dir_path = os.path.abspath(os.path.dirname(__file__))
dir_path = os.path.join(dir_path, "../data/falls/")
directory = os.fsencode(dir_path)
df = None
for file in os.listdir(directory):
  filename = os.fsdecode(file)
  n_frames, rect_h, rect_w, ellipse_angles = processVideo(os.path.join(dir_path, filename))
  current_df = getStatsForVideo(n_frames, rect_h, rect_w, ellipse_angles, filename)
  df = current_df if df is None else df.append(current_df, ignore_index=True)
print(df)
# TODO - df.to_csv('path_of_csv.csv')

## TODO - nonfall videos

