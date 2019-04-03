'''
Given the path to a directory containing videos and an output directory path, 
this splits those videos into 60 frame clips with a stride of 30 frames
and saves them each as their own mp4 file in the given output directory
'''

import os, cv2, sys, getopt, math
import numpy as np
from pathlib import Path

FOUR_CC = cv2.VideoWriter_fourcc(*'mp4v')
FPS = 30.0

def splitVid(parent_reel, out_path, prefix, ext, sub_vid_size=60, stride=30):
  n_frames = int(parent_reel.get(cv2.CAP_PROP_FRAME_COUNT))
  n_subreels = 1
  if (n_frames > sub_vid_size):
    n_subreels = int(n_frames / stride) - 1 if(n_frames % stride == 0) else int(n_frames/stride)
  start_frame_last_subreel = n_frames - sub_vid_size if n_subreels > 1 else 0
  # print('n_frames: {0} | n_subreels: {1} | start frame last subreel: {2}'.format(n_frames, n_subreels, start_frame_last_subreel))
  w = int(parent_reel.get(cv2.CAP_PROP_FRAME_WIDTH))
  h = int(parent_reel.get(cv2.CAP_PROP_FRAME_HEIGHT))
  curr_subreel = 1

  while(curr_subreel <= n_subreels):
    subname = prefix + '-' + str(curr_subreel) + '.' + ext
    sub_full_path = os.path.join(out_path, subname)
    # print('curr subreel: {0} | name of curr subreel: {1}'.format(curr_subreel, subname))
    out = cv2.VideoWriter(sub_full_path, FOUR_CC, FPS, (w,h))
    
    curr_frame_cnt = 1
    while(curr_frame_cnt <= sub_vid_size and curr_frame_cnt <= n_frames):
      # print('curr frame cnt', curr_frame_cnt)
      _, frame = parent_reel.read()
      out.write(frame)
      curr_frame_cnt += 1

    ### Update starting frame for next subreel  
    # If we're on the last subreel, make sure starting frame is sub_vid_size frames before end of video. (will have overlap)
    curr_subreel += 1
    if(curr_subreel == n_subreels):
      # print('curr subreel = {0}, n subreels = {1}, setting next frame to {2}'.format(
      #     curr_subreel, n_subreels, start_frame_last_subreel))
      parent_reel.set(cv2.CAP_PROP_POS_FRAMES, start_frame_last_subreel)
    else:
      # print('setting next frame to {0}'.format(stride * curr_subreel-1))
      parent_reel.set(cv2.CAP_PROP_POS_FRAMES, stride * curr_subreel-1)
    
    out.release()
    print('--------------- End of {0} ----------------'.format(subname))
  print('------------------------------- End of {0} ------------------------------'.format(prefix))

def splitVidsInDirectory(src_path, out_path):
  directory = os.fsencode(src_path)
  for file in os.listdir(directory):
    filename = os.fsdecode(file)
    file_path = str(src_path) + '/' + filename
    reel = cv2.VideoCapture(file_path)
    prefix, ext = filename.split('.')  # Assumes only one '.' in filename
    splitVid(reel, out_path, prefix, ext)
    reel.release()

def parseArgs(argv):
  inputdir = ''
  outputdir = ''
  try:
    opts, args = getopt.getopt(argv, "hi:o:", ["indir=", "outdir="])
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
