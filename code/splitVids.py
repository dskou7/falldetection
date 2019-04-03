'''
Given the path to a directory containing videos and an output directory path, 
this splits those videos into 60 frame clips with a stride of 30 frames
and saves them each as their own mp4 file in the given output directory
'''

import os, cv2, sys, getopt, math
import numpy as np
from pathlib import Path

FOUR_CC = cv2.VideoWriter_fourcc(*'MP4V')
FPS = 30.0

def splitVid(parent_reel, out_path, prefix, ext, sub_vid_size=60, stride=30):
  n_frames = int(parent_reel.get(cv2.CAP_PROP_FRAME_COUNT))
  n_subreels = 1 if n_frames <= sub_vid_size else math.ceil(n_frames / stride)
  start_frame_last_subreel = sub_vid_size - (n_frames % sub_vid_size) if n_subreels > 1 else 0
  w = int(parent_reel.get(cv2.CAP_PROP_FRAME_WIDTH))
  h = int(parent_reel.get(cv2.CAP_PROP_FRAME_HEIGHT))
  subreel_i = 0

  while(subreel_i < n_subreels):
    # _, frame = parent_reel.read()
    subname = prefix + '-' + str(subreel_i+1) + '.' + ext
    sub_full_path = os.path.join(out_path, subname)
    out = cv2.VideoWriter(sub_full_path, FOUR_CC, FPS, (w,h))
    # If we're on the last subreel, make sure starting frame is sub_vid_size before end of frame. (will have overlap)
    if(subreel_i == n_subreels):
      parent_reel.set(cv2.CAP_PROP_POS_FRAMES, start_frame_last_subreel)
    else:
      parent_reel.set(cv2.CAP_PROP_POS_FRAMES, stride * subreel_i)
    
    frame_cnt = 0
    while(frame_cnt < sub_vid_size and frame_cnt < n_frames):
      _, frame = parent_reel.read()
      out.write(frame)
      frame_cnt += 1
    
    out.release()
    subreel_i += 1
  # cv2.destroyAllWindows()

def splitVidsInDirectory(src_path, out_path):
  directory = os.fsencode(src_path)
  for file in os.listdir(directory):
    filename = os.fsdecode(file)
    file_path = str(src_path) + '/' + filename
    reel = cv2.VideoCapture(file_path)
    prefix, ext = filename.split('.')  # Assumes only one '.' in filename
    splitVid(reel, out_path, prefix, ext)

def parseArgs(argv):
  inputdir = ''
  outputdir = ''
  try:
    opts, args = getopt.getopt(argv, "hi:o:", ["idir=", "odir="])
  except getopt.GetoptError:
    print('splitVids.py -i <inputdir> -o <outputdir>')
    sys.exit(2)
  for opt, arg in opts:
    if(opt == '-h'):
      print('splitVids.py -i <inputdir> -o <outputdir>')
      sys.exit()
    elif(opt in ("-i", "--idir")):
      inputdir = Path(arg).resolve()
    elif(opt in ("-o", "--odir")):
      outputdir = Path(arg).resolve()
  if(inputdir == '' or outputdir == ''):
    print('splitVids.py -i <inputdir> -o <outputdir>')
    sys.exit(2)
  return inputdir, outputdir

def main(argv):
  inputdir, outputdir = parseArgs(argv)
  if not os.path.exists(outputdir):
    os.makedirs(outputdir)
  splitVidsInDirectory(inputdir, outputdir)

if (__name__ == "__main__"):
  main(sys.argv[1:])
