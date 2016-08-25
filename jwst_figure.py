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
#import asciitable
import scipy.ndimage
import scipy.stats as ss
import scipy.signal
import scipy as sp
import scipy.odr as odr
import glob
import os
#import make_color_image
#import make_fake_wht
#import gzip
#import tarfile
#import shutil
#import cosmocalc
#import congrid
import astropy.io.ascii as ascii
#import sunpy__load
#import sunpy__plot
#import sunpy__synthetic_image
#from sunpy.sunpy__plot import *
import warnings
import subprocess
#import photutils
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import *
import astropy.io.fits as pyfits
#import statmorph
import datetime



if __name__=="__main__":
    bbf = 'broadbandz.fits'

    bb = pyfits.open(bbf)
    camera = "CAMERA3-BROADBAND"

    bi = 2
    vi = 3
    zi = 6
    ji = 8
    hi = 10
    nc150i = 20
    nc200i = 21
    nc277i = 22
    nc356i = 23
    nc444i = 24
    m770i = 26


    fig = pyplot.figure(figsize=(6.0,2.0), dpi=600)
    pyplot.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0,wspace=0.0,hspace=0.0)
    nx = 6
    ny = 2
    totalcount=0
    fili = [bi,zi,hi,nc200i,nc356i,m770i]

    for i in range(nx):
        
        axi = fig.add_subplot(nx,ny,totalcount+1) 
        axi.set_xticks([]) ; axi.set_yticks([])

        #plot grayscale galaxy image
        data = bb[camera].data[fili[i],:,:]

        norm = ImageNormalize(stretch=LogStretch(),vmin=0.01*np.max(data),vmax=np.max(data),clip=True)
        axi.imshow(data, origin='lower', cmap='Greys_r', norm=norm, interpolation='nearest')
        #axi.annotate('{:3.2f}$\mu m$'.format(image_hdu.header['EFLAMBDA']),xy=(0.05,0.05),xycoords='axes fraction',color='white',ha='left',va='center',size=6)

        totalcount = totalcount+1


    fig.savefig('jwst.pdf',dpi=600)
    pyplot.close(fig)
