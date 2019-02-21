# Author: Ankush Gupta
# Date: 2015

"""
Entry-point for generating synthetic text images, as described in:

@InProceedings{Gupta16,
      author       = "Gupta, A. and Vedaldi, A. and Zisserman, A.",
      title        = "Synthetic Data for Text Localisation in Natural Images",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
      year         = "2016",
    }
"""

import numpy as np
import h5py
import os, sys, traceback
import os.path as osp
from synthgen import *
from common import *
import wget, tarfile


## Define some configuration variables:
NUM_IMG = -1 # no. of images to use for generation (-1 to use all available):
SECS_PER_IMG = 10000000 #max time per image in seconds

# path to the data-file, containing image, depth and segmentation:
DATA_PATH = 'data'
DB_FNAME = osp.join(DATA_PATH,'dset.h5')
# url of the data (google-drive public file):
DATA_URL = 'http://www.robots.ox.ac.uk/~ankush/data.tar.gz'

def get_data():
  """
  Download the image,depth and segmentation data:
  Returns, the h5 database.
  """
  if not osp.exists(DB_FNAME):
    try:
      colorprint(Color.BLUE,'\tdownloading data (56 M) from: '+DATA_URL,bold=True)
      print
      sys.stdout.flush()
      out_fname = 'data.tar.gz'
      wget.download(DATA_URL,out=out_fname)
      tar = tarfile.open(out_fname)
      tar.extractall()
      tar.close()
      os.remove(out_fname)
      colorprint(Color.BLUE,'\n\tdata saved at:'+DB_FNAME,bold=True)
      sys.stdout.flush()
    except:
      print colorize(Color.RED,'Data not found and have problems downloading.',bold=True)
      sys.stdout.flush()
      sys.exit(-1)
  # open the h5 file and return:
  return h5py.File(DB_FNAME,'r')


def add_res_to_db(imgname,res,db):
  """
  Add the synthetically generated text image instance
  and other metadata to the dataset.
  """
  ninstance = len(res)
  for i in xrange(ninstance):
    dname = "%s_%d"%(imgname, i)
    db['data'].create_dataset(dname,data=res[i]['img'])
    db['data'][dname].attrs['charBB'] = res[i]['charBB']
    db['data'][dname].attrs['wordBB'] = res[i]['wordBB']        
    db['data'][dname].attrs['txt'] = res[i]['txt']


def main(viz=False):
  # open databases:
  print colorize(Color.BLUE,'getting data..',bold=True)
  #db = get_data()
  db = h5py.File('receipt.h5', 'r')
  print colorize(Color.BLUE,'\t-> done',bold=True)

  # open the output h5 file:
  out_db = h5py.File(args.out_file,'w')
  out_db.create_group('/data')
  print colorize(Color.GREEN,'Storing the output in: '+args.out_file, bold=True)

  # get the names of the image files in the dataset:
  imnames = sorted(db['image'].keys())
  N = len(imnames)
  global NUM_IMG
  if NUM_IMG < 0:
    NUM_IMG = N
  start_idx,end_idx = 0,min(NUM_IMG, N)

  RV3 = RendererV3(DATA_PATH,max_time=SECS_PER_IMG)

  override = [0, 2, 4, 6, 7,8,9,10,11,12,13,14,16,17,18,20]
  for i in override:#xrange(start_idx,end_idx):
    imname = imnames[i]
    try:
      # get the image:
      img = Image.fromarray(db['image'][imname][:])
      # get the pre-computed depth:
      #  there are 2 estimates of depth (represented as 2 "channels")
      #  here we are using the second one (in some cases it might be
      #  useful to use the other one):
      depth = db['depth'][imname][:]
      #depth = depth[:,:,1]
      # get segmentation:
      seg = db['seg'][imname][:].astype('float32')
      area = db['seg'][imname].attrs['area']
      label = db['seg'][imname].attrs['label']

      img = np.array(img)
      seg = np.array(seg)
      # re-size uniformly:
#      sz = depth.shape[:2][::-1]
#      img = np.array(img.resize(sz,Image.ANTIALIAS))
#      seg = np.array(Image.fromarray(seg).resize(sz,Image.NEAREST))

      print colorize(Color.RED,'%d of %d'%(i,end_idx-1), bold=True)
      res = RV3.render_text(img,depth,seg,area,label,
                            ninstance=args.num_reuse,viz=viz)
      if len(res) > 0:
        # non-empty : successful in placing text:
        add_res_to_db(imname,res,out_db)
      # visualize the output:
      if viz:
        if 'q' in raw_input(colorize(Color.RED,'continue? (enter to continue, q to exit): ',True)):
          break
    except:
      traceback.print_exc()
      print colorize(Color.GREEN,'>>>> CONTINUING....', bold=True)
      continue
  db.close()
  out_db.close()


if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
  parser.add_argument('--viz',action='store_true',dest='viz',default=False,help='flag for turning on visualizations')
  parser.add_argument('--num-reuse', type=int, default=1)
  parser.add_argument('--out-file', type=str)
  args = parser.parse_args()
  main(args.viz)
