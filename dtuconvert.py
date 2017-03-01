import astropy.io.fits as pyfits
import math
import matplotlib
import matplotlib.pyplot as pyplot
import numpy as np
import scipy.ndimage
import scipy as sp

import glob

import congrid

import os

import time


def convert_galaxy(filename,outfilename):
    print(filename,outfilename)
    array = pyfits.open(filename)[0].data
    new_array = congrid.congrid(array,(64,64))
    
    fu = pyfits.PrimaryHDU(new_array)
    ful = pyfits.HDUList([fu])

    ful.writeto(outfilename,overwrite=True)


    return



def convert_galaxies(sd='subdir_010'):
    image_base='/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/snapshot_068/'

    

    files_105 = np.sort(np.asarray(glob.glob(image_base+str(sd)+'/images_subhalo_*/*WFC3-F105W_SB00.fits')))
    files_125 = np.sort(np.asarray(glob.glob(image_base+str(sd)+'/images_subhalo_*/*WFC3-F125W_SB00.fits')))
    files_160 = np.sort(np.asarray(glob.glob(image_base+str(sd)+'/images_subhalo_*/*WFC3-F160W_SB00.fits')))


    print(files_125.shape)

    if not os.path.lexists(sd):
        os.mkdir(sd)

    for f in files_105:
        outfilename=sd+'/rebin_'+os.path.basename(f)
        convert_galaxy(f,outfilename)
 
    for f in files_125:
        outfilename=sd+'/rebin_'+os.path.basename(f)
        convert_galaxy(f,outfilename)
        
    for f in files_160:
        outfilename=sd+'/rebin_'+os.path.basename(f)
        convert_galaxy(f,outfilename)
               

    return



if __name__=="__main__":


    st = time.time()

    convert_galaxies(sd='subdir_001')
    convert_galaxies(sd='subdir_002')
    convert_galaxies(sd='subdir_003')
    convert_galaxies(sd='subdir_004')
    convert_galaxies(sd='subdir_005')

    convert_galaxies(sd='subdir_006')
    convert_galaxies(sd='subdir_007')
    convert_galaxies(sd='subdir_008')
    convert_galaxies(sd='subdir_009')
    convert_galaxies(sd='subdir_010')

    convert_galaxies(sd='subdir_011')
    convert_galaxies(sd='subdir_012')
    convert_galaxies(sd='subdir_013')
    convert_galaxies(sd='subdir_014')
    convert_galaxies(sd='subdir_015')

    convert_galaxies(sd='subdir_016')
    convert_galaxies(sd='subdir_017')
    convert_galaxies(sd='subdir_018')
    convert_galaxies(sd='subdir_019')
    convert_galaxies(sd='subdir_020')

    convert_galaxies(sd='subdir_021')
    convert_galaxies(sd='subdir_022')



    et = time.time()

    print(et-st)
