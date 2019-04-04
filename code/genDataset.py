import os, cv2, sys, getopt
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
from sklearn import preprocessing


############## Function declarations ##############
def processVideo(full_path):
  reel = cv2.VideoCapture(full_path)
  ## Make background subtractor
  fgbg = cv2.createBackgroundSubtractorMOG2()
  ## Initialize variables
  n_frames = int(reel.get(cv2.CAP_PROP_FRAME_COUNT))
  rect_w = np.zeros(n_frames)
  rect_h = np.zeros(n_frames)
  ellipse_angle = np.zeros(n_frames)

  curr_frame_i = 0
  while(curr_frame_i < n_frames):
    _, frame = reel.read()
    fgmask = fgbg.apply(frame)

    ## TODO - replace this with MXNET human detection
    contours, _ = cv2.findContours(
        fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if(len(contours) == 0):
      print('No contours found in frame {0}, marking as outlier, continuing to next frame'.format(curr_frame_i))
      ellipse_angle[curr_frame_i] = rect_w[curr_frame_i] = rect_h[curr_frame_i] = -999999
      curr_frame_i += 1
      continue
    x, y, w, h = cv2.boundingRect(contours[0])
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # fitEllipse throws an error if the contour doesn't have at least 5 points. This occurs on first frame often (just noise)
    if(len(contours[0]) >= 5):
      ellipse = cv2.fitEllipse(contours[0])
      cv2.ellipse(frame, ellipse, (0, 0, 255), 2)
      ellipse_angle[curr_frame_i] = ellipse[2]
    rect_w[curr_frame_i] = w
    rect_h[curr_frame_i] = h
    # print('curr_frame_i:', curr_frame_i)
    # print('rect_ratio: {0} | ellipse_angle: {1} '.format(rect_ratio[curr_frame_i], ellipse_angle[curr_frame_i]))
    curr_frame_i += 1

  reel.release()
  cv2.destroyAllWindows()
  return n_frames, rect_h, rect_w, ellipse_angle


def removeOutliers(x, outlierConstant=1.5):
    upper_quartile = np.percentile(x, 75)
    lower_quartile = np.percentile(x, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    result = x[np.where(
        (x >= quartileSet[0]) & (x <= quartileSet[1]))]
    return result

# moving average: calculate average of given number (window len) of neighbors
def avg_smooth(x, window_len=13):
  df = pd.DataFrame(x)
  return df.rolling(window_len).mean()

# Uses the Savitzky-Golay filter (finds least-square fit). Good for signal data
####### SEEMS SUPERIOR TO AVG_SMOOTH
def sg_smooth(x, window_len=13):
  return savgol_filter(x, window_len, 3)

def clean(x, smoother=sg_smooth):
  x = removeOutliers(x)
  return smoother(x)

'''
Calculates the biggest change that occurs in a variable over time, after removing outliers. 
@param x: an np array representing the variable over time
@param outlierConstant: The factor for how far out from the IQR a value must be to be considered an outlier
@return: the change in the variable. Positive means it increased with time, negative it decreased with time
'''
# If the min is at an earlier frame than the max, the variable has increased so direction should be positive.
# RISK - but what if there are multiple mins/maxes throughout the video? argmin/argmax only returns the first occurence
# RISK 2 (CRITICAL) - the biggest change may be an uninteresting one (IE the biggest change in h for a given clip may be negative, which we don't care about) but that doesn't mean the second biggest change wouldn't be a significant interesting one. This is a problem
def calcBiggestChange(x):
  min_i = np.argmin(x)
  max_i = np.argmax(x)
  direction = 1 if min_i < max_i else -1
  delta = direction * (x[max_i] - x[min_i])
  return delta

# Title will just be y_label concatenated with ' Over Time'
def plotVariableVsFrame(variable, y_label):
  plt.scatter(range(len(variable)), variable)
  plt.title(y_label + ' Over Time')
  plt.xlabel('Video Frame')
  plt.ylabel(y_label)
  plt.show()

def visualizeSmoothing(n_frames, rect_h, rect_w, ellipse_angle, rect_ratio):
  plotVariableVsFrame(rect_h, 'Rectangle Height')
  clean_h = clean(rect_h, sg_smooth)
  plotVariableVsFrame(clean_h, 'Cleansed height with sg_smooth')
  clean_h = clean(rect_h, avg_smooth)
  plotVariableVsFrame(clean_h, 'Cleansed height with avg_smooth')
  plotVariableVsFrame(rect_w, 'Rectangle Width')
  clean_w = clean(rect_w, sg_smooth)
  plotVariableVsFrame(clean_w, 'Cleansed width with sg_smooth')
  clean_w = clean(rect_w, avg_smooth)
  plotVariableVsFrame(clean_w, 'Cleansed width with avg_smooth')
  plotVariableVsFrame(rect_ratio, 'Rectangle Ratio')
  clean_ratio = clean(rect_ratio, sg_smooth)
  plotVariableVsFrame(clean_ratio, 'Cleansed ratio with sg_smooth')
  clean_ratio = clean(rect_ratio, avg_smooth)
  plotVariableVsFrame(clean_ratio, 'Cleansed ratio with avg_smooth')
  plotVariableVsFrame(ellipse_angle, 'Ellipse Angle')
  clean_angle = clean(ellipse_angle, sg_smooth)
  plotVariableVsFrame(clean_angle, 'Cleansed angle with sg_smooth')
  clean_angle = clean(ellipse_angle, avg_smooth)
  plotVariableVsFrame(clean_angle, 'Cleansed angle with avg_smooth')

def getStatsForVideo(n_frames, rect_h, rect_w, ellipse_angle, video_name, label):
  # Clean the inputs, calculate stats
  rect_ratio = rect_w / rect_h
  rect_h, rect_w, ellipse_angle, rect_ratio = clean(rect_h), clean(rect_w), clean(ellipse_angle), clean(rect_ratio)
  delta_h, delta_w, delta_angle, delta_ratio = calcBiggestChange(rect_h), calcBiggestChange(
      rect_w), calcBiggestChange(ellipse_angle), calcBiggestChange(rect_ratio)

  # Make sure smoothing is working as intended
  # visualizeSmoothing(n_frames, rect_h, rect_w, ellipse_angle, rect_ratio)

  # Save stats to a df row
  data = {'Video': [video_name], 'Label': [label], 'Delta h': [delta_h], 'Delta w': [delta_w], 'Delta ratio': [delta_ratio], 'Delta angle': [delta_angle] }
  df = pd.DataFrame(data, columns=['Video', 'Label', 'Delta h', 'Delta w', 'Delta ratio', 'Delta angle'])
  return df



'''
Loops through the fall or nonfall directory of data videos and generates a data frame representing the feature set for the videos
@param fall: boolean 
'''
def processVideosForClass(dir_path, label, df=None):
  directory = os.fsencode(dir_path)
  for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if(filename[0] == '.'):
      print('Skipping hidden file {0}'.format(filename))
      continue
    print('Processing video {0}'.format(filename))
    n_frames, rect_h, rect_w, ellipse_angle = processVideo(os.path.join(dir_path, filename))
    current_df = getStatsForVideo(n_frames, rect_h, rect_w, ellipse_angle, filename, label)
    df = current_df if df is None else df.append(current_df, ignore_index=True)
  return df


def parseArgs(argv):
  fall_dir = ''
  nonfall_dir = ''
  outfile = ''
  try:
    opts, args = getopt.getopt(argv, "hf:n:o:", ['fall-dir=', 'nonfall-dir=', 'outfile=']) # short opts w/ ':' after and long opts with '=' after are required
  except getopt.GetoptError:
    print('genDataset.py -f <fall-dir> -n <nonfall-dir> -o <outfile>')
    sys.exit(2)
  for opt, arg in opts:
    if(opt == '-h'):
      print('genDataset.py -f <fall-dir> -n <nonfall-dir> -o <outfile>')
      sys.exit()
    elif(opt in ("-f", "--fall-dir")):
      fall_dir = Path(arg).resolve()
    elif(opt in ("-n", "--nonfall-dir")):
      nonfall_dir = Path(arg).resolve()
    elif(opt in ("-o", "--outfile")):
      outfile = Path(arg).resolve()
  if(fall_dir == '' or nonfall_dir == '' or outfile == ''):
    print('genDataset.py -f <fall-dir> -n <nonfall-dir> -o <outfile>')
    sys.exit(2)
  return fall_dir, nonfall_dir, outfile

def main(argv):
  fall_dir, nonfall_dir, outfile = parseArgs(argv)
  df = processVideosForClass(nonfall_dir, 0)
  df = processVideosForClass(fall_dir, 1, df=df)
  # print(df)
  ## Normalize continuous columns
  x = df[['Delta h', 'Delta w', 'Delta ratio', 'Delta angle']].values
  x_scaled = preprocessing.normalize(x, norm='l1')
  df[['Delta h', 'Delta w', 'Delta ratio', 'Delta angle']] = pd.DataFrame(x_scaled)
  print(df)
  df.to_csv(outfile, index=False)

  ### Uncomment this to test things on just one video and comment the rest of main out. TODO - outdated
  # dir_path = os.path.abspath(os.path.dirname(__file__))
  # n_frames, rect_h, rect_w, ellipse_angle = processVideo(os.path.join(dir_path, "../data/fall/fall1.mp4"))
  # current_df = getStatsForVideo(n_frames, rect_h, rect_w, ellipse_angle, "../data/fall/fall1.mp4", 1)
  # print(current_df)

if (__name__ == "__main__"):
  main(sys.argv[1:])




