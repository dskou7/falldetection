'''
Given the path to a directory containing videos and an output directory path, 
this splits those videos into 60 frame clips with a stride of 30 frames
and saves them each as their own mp4 file in the given output directory
'''

import os, cv2, sys, getopt, math
import numpy as np
from pathlib import Path

def splitVid(reel, filename, sub_vid_size=60, stride=30):
  n_frames = int(reel.get(cv2.CAP_PROP_FRAME_COUNT))
  subreels = np.zeros(math.ceil(n_frames / sub_vid_size))
  print('n_frames for {0}: {1}'.format(filename, n_frames))
  print('num subreels for {0}: {1}'.format(filename, len(subreels)))
  return subreels

def splitVidsInDirectory(src_path, out_path):
  directory = os.fsencode(src_path)
  out_vids = []
  outvid_names = []
  for file in os.listdir(directory):
    filename = os.fsdecode(file)
    reel = cv2.VideoCapture(str(src_path) + '/' + filename)
    sub_vids = splitVid(reel, filename)
    prefix = filename.split('.')[0]
    outvid_names += [prefix + '-' + str(i+1) + '.mp4' for i in range(len(sub_vids))]
    out_vids += [sub_vids]
  print('outvid names:', outvid_names)

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
  splitVidsInDirectory(inputdir, outputdir)

if (__name__ == "__main__"):
  main(sys.argv[1:])
