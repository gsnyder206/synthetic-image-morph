import cProfile
import pstats
import math
import string
import sys
import struct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
import matplotlib.colors as pycolors
import matplotlib.cm as cm
import matplotlib.patches as patches
import numpy as np
import cPickle
import asciitable
import scipy.ndimage
import scipy.stats as ss
import scipy.signal
import scipy as sp
import scipy.odr as odr
import glob
import os
import make_color_image
import make_fake_wht
import gzip
import tarfile
import shutil
import cosmocalc
import congrid
import astropy.io.ascii as ascii
import sunpy__load
import sunpy__plot
import sunpy__synthetic_image
from sunpy.sunpy__plot import *
import warnings
import subprocess
import photutils
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import *
import astropy.io.fits as pyfits
import statmorph
import datetime
import h5py
import requests
from multiprocessing import Process, Queue, current_process
import time


def analyze_morphology(gbandfile,whiteseg):


    

    return


def worker(input,output,**kwargs):
    for func, args in iter(input.get,'STOP'):
        f = calculate(func,args,**kwargs)
        output.put(f)

def calculate(func,args,**kwargs):
    result = func(*args,**kwargs)
    #return '%s was given %s and got %s' % \
    #         (current_process().name, args, result)
    return result






def process_single_object():

    #do a check if segmap already exists.  if so, just run morphology (start here for tests)


    return



def process_directory():

    #assign individual objects into separate processes

    segs = np.sort(np.asarray(glob.glob('*_white_cold_seg.fits')))
    

    return 0


def do_nonmerger_test():
    analysis_dir = "/home/gsnyder/oasis_project/PanSTARRS/nonmergers"
    result = process_directory(analysis_dir)
    return

    
if __name__=="__main__":


    #cProfile.run('test_10107()','profiler_stats_10107')
    #p = pstats.Stats('profiler_stats_10107')
    #p.strip_dirs().sort_stats('time').print_stats(45)

    cProfile.run('do_nonmerger_test()','profiler_stats_nonmerger_test')
    p = pstats.Stats('profiler_stats_nonmerger_test')
    p.strip_dirs().sort_stats('time').print_stats(45)
    
