import cProfile
import pstats
import math
import string
import sys
import struct
import matplotlib
matplotlib.use('Agg')
import numpy as np
import scipy.ndimage
import scipy.stats as ss
import scipy.signal
import scipy as sp
import scipy.odr as odr
import glob
import os
import gzip
import tarfile
import shutil
import congrid
import astropy.io.ascii as ascii
import warnings
import subprocess
import photutils
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import *
import astropy.io.fits as pyfits
#import statmorph
import datetime
import setup_synthetic_images_mp as ssimp

# Based on candelize.py

def process_snapshot(subdirpath='.', clobber=False, galaxy=None,
        seg_filter_label='ps1_i', magsb_limits=[21, 22, 23, 24, 25],
        camindices=[0,1,2,3], do_idl=False, analyze=True, use_nonscatter=True, Np=4):

    cwd = os.path.abspath(os.curdir)

    os.chdir(subdirpath)

    bbfile_list = np.sort(np.asarray(glob.glob('broadbandz.fits*')))   #enable reading .fits.gz files
    print('bbfile_list =')
    print(bbfile_list)

    if galaxy is not None:
        thisbb = np.where(bbfile_list==galaxy)[0]
        bbfile_list= bbfile_list[thisbb]

    test_file = bbfile_list[0]
    tf = pyfits.open(test_file)
    print(tf.info())
    print(tf['BROADBAND'].header.cards)
    print(tf['SFRHIST'].header.get('star_adaptive_smoothing'))
    print(tf['SFRHIST'].header.get('star_radius_factor'))

    #this is critical for later
    
    fils = tf['FILTERS'].data.field('filter')
    print(fils)

    tf.close()

    # Ignore GALEX for now.
    # Data for later: 1.5 arcsec per pixel, ~25 sbmag limit,
    # fwhm = 4.0, 5.6 arcsec for fuv and nuv, respectively

    filters_to_analyze = [
            'panstarrs/panstarrs_ps1_g',
            'panstarrs/panstarrs_ps1_r',
            'panstarrs/panstarrs_ps1_w',
            'panstarrs/panstarrs_ps1_open',
            'panstarrs/panstarrs_ps1_i',
            'panstarrs/panstarrs_ps1_z',
            'panstarrs/panstarrs_ps1_y']

    skip_filter_boolean = [
            False,
            False,
            False,
            False,
            False,
            False,
            False]

    print(filters_to_analyze)

    # Pixel size in arcsec.
    pixsize_arcsec = [
            0.262,
            0.262,
            0.262,
            0.262,
            0.262,
            0.262,
            0.262]
    
    filter_labels = [
            'ps1_g',
            'ps1_r',
            'ps1_w',
            'ps1_open',
            'ps1_i',
            'ps1_z',
            'ps1_y']

    filter_indices = []

    print(len(filters_to_analyze), len(skip_filter_boolean), len(filter_labels))
    
    for i,f in enumerate(filters_to_analyze):
        fi = np.where(fils==f)
        print(fi[0][0], f, fils[fi[0][0]], filter_labels[i]) #, filters_to_analyze[fi]
        filter_indices.append(fi[0][0])

    filter_indices = np.asarray(filter_indices)

    print(filter_indices)

    # order of filter_labels in wavelength space
    filter_lambda_order = [0, 1, 2, 3, 4, 5, 6]

    #photfnu units Jy; flux in 1 ct/s
    photfnu_Jy = [
            2.14856e-07,
            1.77931e-07,
            5.49429e-08,
            4.06004e-08,
            1.81461e-07,
            2.65602e-07,
            6.62502e-07]
    
    morphcode_dir = "/Users/gsnyder/Documents/pro/morph_december2013/morph_pro/"
    morphcode_files = np.asarray(glob.glob(os.path.join(morphcode_dir,"*.*")))

    se_dir = '/Users/gsnyder/Documents/Projects/Illustris_Morphology/Illustris-CANDELS/SE_scripts'
    se_files = np.asarray(glob.glob(os.path.join(se_dir,"*.*")))

    # Use custom-made gaussian PSF (cannot find the actual PSF)
    psf_dir = '/home/vrg/filter_data/psf'
    psf_names = ['%s.fits' % (f) for f in filters_to_analyze]

    # A bit of oversampling:
    psf_pix_arcsec = [0.262, 0.262, 0.262, 0.262, 0.262, 0.262, 0.262]
    psf_truncate = [None, None, None, None, None, None, None]
    psf_hdu_num = [0, 0, 0, 0, 0, 0, 0]
    psf_fwhm = [1.31, 1.19, 1.31, 1.31, 1.11, 1.07, 1.02]  # in arcsec

    psf_files = []
    for pname in psf_names:
        psf_file = os.path.join(psf_dir,pname)
        psf_files.append(psf_file)
        print(psf_file, os.path.lexists(psf_file))

    ###  PSFSTD; WFC3 = 0.06 arcsec, ACS = 0.03 arcsec... I think
    ### NIRCAM in header with keyword 'PIXELSCL';  short 0.07925 long 0.0162
    ## acs wfc 0.05 arcsec pixels... PSFSTD x4 oversample?
    ## wfc3 ir 0.13 arcsec
    ## wfc3 uv 0.04 arcsec

    mockimage_parameters = ssimp.analysis_parameters('mockimage_default')
    mockimage_parameters.filter_indices = filter_indices
    mockimage_parameters.filter_labels = filter_labels
    mockimage_parameters.pixsize_arcsec = pixsize_arcsec
    mockimage_parameters.morphcode_base = morphcode_dir
    mockimage_parameters.morphcode_files = morphcode_files
    mockimage_parameters.se_base = se_dir
    mockimage_parameters.se_files = se_files
    mockimage_parameters.camera_indices = camindices #None #by default, do all
    mockimage_parameters.psf_files = psf_files
    mockimage_parameters.psf_pix_arcsec = psf_pix_arcsec
    mockimage_parameters.psf_truncate = psf_truncate
    mockimage_parameters.psf_hdu_num = psf_hdu_num
    mockimage_parameters.magsb_limits = magsb_limits
    mockimage_parameters.psf_fwhm_arcsec = psf_fwhm
    mockimage_parameters.photfnu_Jy = photfnu_Jy
    mockimage_parameters.filter_lambda_order = filter_lambda_order
    mockimage_parameters.skip_filters = skip_filter_boolean
    mockimage_parameters.use_nonscatter = use_nonscatter
    
    #use exactly one detection and segmentation per object, depending on redshift
    #enormous simplification
    #observationally, go w deepest filter.  here... ?

    print(seg_filter_label)
    print(mockimage_parameters.filter_labels)

    mockimage_parameters.segment_filter_label = seg_filter_label
    mockimage_parameters.segment_filter_index = np.where(np.asarray(mockimage_parameters.filter_labels) == seg_filter_label)[0][0]

    print(mockimage_parameters.segment_filter_label)
    print(mockimage_parameters.segment_filter_index)
    
    assert(len(psf_pix_arcsec)==len(pixsize_arcsec))
    assert(len(filter_labels)==len(mockimage_parameters.psf_files))

    bbdirs = []
    
    for i,bbfile in enumerate(bbfile_list):

        bbdir = ssimp.process_single_broadband(bbfile, mockimage_parameters,
                clobber=clobber, do_idl=do_idl, analyze=analyze,
                bbase="broadbandz", Np=Np, zip_after=False)
        bbdirs.append(bbdir)
        #~ try:
            #~ bbdir = ssimp.process_single_broadband(bbfile, mockimage_parameters,
                    #~ clobber=clobber, do_idl=do_idl, analyze=analyze,
                    #~ bbase="broadbandz", Np=Np)
            #~ bbdirs.append(bbdir)
        #~ except (KeyboardInterrupt,NameError,AttributeError,KeyError,TypeError,IndexError) as e:
            #~ print(e)
            #~ raise
        #~ except:
            #~ print("Exception while processing broadband: ", bbfile)
            #~ print("Error:", sys.exc_info()[0])
        #~ else:
            #~ print("Successfully processed broadband: ", bbfile)

    os.chdir(cwd)

    return bbdirs



if __name__=="__main__":
    
    # The 5 sigma depths in ABmags are 23.3, 23.2, 23.1, 22.3, 21.3 (grizy filters).
    # For consistency with the rest of the code, we round these numbers.
    # We also include a value of 24, for comparison purposes.
    
    # Without dust
    res = process_snapshot(subdirpath='.', seg_filter_label='ps1_i',
            magsb_limits=[21, 22, 23, 24, 25], camindices=[0,1,2,3],
            do_idl=False, analyze=True, use_nonscatter=True, Np=4)
    #~ # Include dust
    #~ res = process_snapshot(subdirpath='.', seg_filter_label='ps1_g',
            #~ magsb_limits=[23.3, 23.2, 23.1, 22.3, 21.3], camindices=[0,1,2,3],
            #~ do_idl=False, analyze=True, use_nonscatter=False, Np=4)

    
