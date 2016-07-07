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
#import sunpy__load
#import sunpy__plot
#import sunpy__synthetic_image
#from sunpy.sunpy__plot import *
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


def analyze_morphology(gbandfile,gwtfile,whiteseg,se_catalog):

    ghdu = pyfits.open(gbandfile)[0]
    wthdu = pyfits.open(gwtfile)[0]
    seghdu = pyfits.open(whiteseg)[0]
    se_cat = ascii.read(se_catalog)

    #decide here which se number to study
    imn = ghdu.data.shape[0]
    #assumes square
    cl = imn/2 -2
    ch = imn/2 +2
    centersegvals = seghdu.data[cl:ch,cl:ch]
    uniqueseg = np.unique(centersegvals)
    n_uniq = uniqueseg.shape[0]
    if not n_uniq==1 or uniqueseg[0]==0:
        print "Unable to find unique central object, skipping: ", (seghdu.fileinfo()['file']).name
    else:
        segi = np.where(se_cat['NUMBER']==uniqueseg[0])[0][0]
        
        #print segi, uniqueseg, n_uniq, se_cat[segi]
            
    #brightest NOT best -- use also class_star and seg at image center!
    
    se_cat = se_cat[segi]  #0 for testing

    #obj = statmorph.galdata()

    result_hdu,newseg_hdu,obj = statmorph.morph_from_panstarrs_image(ghdu,wthdu,seghdu,se_cat)
    #obj.gfile = gbandfile
    #obj.wtfile = gwtfile
    #obj.whiteseg = whiteseg
    
    return obj


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



def process_directory(directory,Np=2,maxq=10000,lim=None):

    #assign individual objects into separate processes
    cwd = os.path.abspath(os.curdir)
    
    os.chdir(directory)
    
    segs = np.sort(np.asarray(glob.glob('*_white_cold_seg.fits')))
    print 'Number of seg files:', segs.shape
    print segs[0]
    N_objects = segs.shape[0]

    NUMBER_OF_PROCESSES=Np
    task_queue = Queue()
    done_queue = Queue()
    TASKS = []
    TASKS_DONE = []
    TASKS_LEFT = []

    if lim is None:
        lim=np.int64(N_objects)


        
    for i,segfile in enumerate(segs[0:lim]):
        base = segfile.rstrip('_white_cold_seg.fits')
        gfile = base+'_g.fits'
        wtfile = base+'_g.wt.fits'
        se_file = base+'_white_cold.cat'

        if not os.path.lexists(gfile) or not os.path.lexists(wtfile) or not os.path.lexists(se_file):
            print "Missing a file, skipping... ", segfile
        else:
            print "Processing... ", gfile, pyfits.open(gfile)[0].data.shape[0]
            task = (analyze_morphology,(gfile,wtfile,segfile,se_file))
            if i <= maxq:
                task_queue.put(task)
                TASKS.append(task)
            else:
                TASKS_LEFT.append(task)


    for p in range(NUMBER_OF_PROCESSES):
        Process(target=worker,args=(task_queue,done_queue)).start()

    finished_objs = []
        
    while len(TASKS_LEFT) > 0:
        finished_objs.append(done_queue.get())
        newtask = TASKS_LEFT.pop()
        task_queue.put(newtask)

    for i in range(min(maxq,lim)):
        finished_objs.append(done_queue.get())

    print len(finished_objs)
    print finished_objs[0]
    

    for p in range(NUMBER_OF_PROCESSES):
        task_queue.put('STOP')



        
    os.chdir(cwd)
    return finished_objs


def do_nonmerger_test(Np=1,lim=None):
    analysis_dir = "/home/gsnyder/oasis_project/PanSTARRS/nonmergers"
    objects = process_directory(analysis_dir,Np=Np,maxq=10000,lim=lim)
    for go in objects:
        print "Finished.. ", go.imagefile
    
    return

    
if __name__=="__main__":


    #cProfile.run('test_10107()','profiler_stats_10107')
    #p = pstats.Stats('profiler_stats_10107')
    #p.strip_dirs().sort_stats('time').print_stats(45)


    
    cProfile.run('do_nonmerger_test(Np=1,lim=1)','profiler_stats_nonmerger_test_1_1')
    p = pstats.Stats('profiler_stats_nonmerger_test_1_1')
    p.strip_dirs().sort_stats('time').print_stats(15)
    
    cProfile.run('do_nonmerger_test(Np=1,lim=10)','profiler_stats_nonmerger_test_1_10')
    p = pstats.Stats('profiler_stats_nonmerger_test_1_10')
    p.strip_dirs().sort_stats('time').print_stats(15)

    cProfile.run('do_nonmerger_test(Np=2,lim=10)','profiler_stats_nonmerger_test_2_10')
    p = pstats.Stats('profiler_stats_nonmerger_test_2_10')
    p.strip_dirs().sort_stats('time').print_stats(15)
    
    cProfile.run('do_nonmerger_test(Np=2,lim=100)','profiler_stats_nonmerger_test_2_100')
    p = pstats.Stats('profiler_stats_nonmerger_test_2_100')
    p.strip_dirs().sort_stats('time').print_stats(15)

    '''
    cProfile.run('do_nonmerger_test(Np=4,lim=100)','profiler_stats_nonmerger_test_4_100')
    p = pstats.Stats('profiler_stats_nonmerger_test_4_100')
    p.strip_dirs().sort_stats('time').print_stats(15)

    cProfile.run('do_nonmerger_test(Np=4,lim=1000)','profiler_stats_nonmerger_test_4_1000')
    p = pstats.Stats('profiler_stats_nonmerger_test_4_1000')
    p.strip_dirs().sort_stats('time').print_stats(15)

    cProfile.run('do_nonmerger_test(Np=8,lim=1000)','profiler_stats_nonmerger_test_8_1000')
    p = pstats.Stats('profiler_stats_nonmerger_test_8_1000')
    p.strip_dirs().sort_stats('time').print_stats(15)

    cProfile.run('do_nonmerger_test(Np=16,lim=1000)','profiler_stats_nonmerger_test_16_1000')
    p = pstats.Stats('profiler_stats_nonmerger_test_16_1000')
    p.strip_dirs().sort_stats('time').print_stats(15)
    '''
