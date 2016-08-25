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
    fils = bb['FILTERS'].data['filter']

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
    lams = [0.45,0.85,1.6,2.0,3.5,7.7]
    origpixkpc = 0.0625
    fwhm_arcsec = np.asarray([0.08,0.12,0.20,0.10,0.20,0.40])
    sigma_arcsec = fwhm_arcsec/2.35
    scale = 6.05 #kpc/arcsec
    sigma_kpc = sigma_arcsec*scale
    sigma_pix = sigma_kpc/origpixkpc

    for i in range(nx):
        
        axi = fig.add_subplot(ny,nx,totalcount+1) 
        axi.set_xticks([]) ; axi.set_yticks([])

        #plot grayscale galaxy image
        data = bb[camera].data[fili[i],350:450,350:450]*(lams[i]**2)
        print fils[fili[i]], np.max(data), 0.01*np.max(data)


        norm = ImageNormalize(stretch=LogStretch(),vmin=0.01,vmax=0.25,clip=True)
        axi.imshow(data, origin='lower', cmap='Greys_r', norm=norm, interpolation='nearest')
        #axi.annotate('{:3.2f}$\mu m$'.format(image_hdu.header['EFLAMBDA']),xy=(0.05,0.05),xycoords='axes fraction',color='white',ha='left',va='center',size=6)

        totalcount = totalcount+1




    #resB = sp.ndimage.filters.gaussian_filter(b,sigma[2],output=sB)

    for i in range(nx):
        
        axi = fig.add_subplot(ny,nx,totalcount+1) 
        axi.set_xticks([]) ; axi.set_yticks([])

        #plot grayscale galaxy image
        data = bb[camera].data[fili[i],350:450,350:450]*(lams[i]**2)
        print fils[fili[i]], np.max(data), 0.01*np.max(data), sigma_pix[i]
        cdata = data*1.0

        resc = sp.ndimage.filters.gaussian_filter(data,sigma_pix[i],output=cdata)
        print fils[fili[i]], np.max(cdata), 0.01*np.max(cdata), sigma_pix[i]

        norm = ImageNormalize(stretch=LogStretch(),vmin=0.01,vmax=0.25,clip=True)
        axi.imshow(cdata*np.sum(data)/np.sum(cdata), origin='lower', cmap='Greys_r', norm=norm, interpolation='nearest')
        #axi.annotate('{:3.2f}$\mu m$'.format(image_hdu.header['EFLAMBDA']),xy=(0.05,0.05),xycoords='axes fraction',color='white',ha='left',va='center',size=6)

        totalcount = totalcount+1



    fig.savefig('jwst.pdf',dpi=600)
    pyplot.close(fig)
