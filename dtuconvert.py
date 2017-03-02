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

import h5py

def convert_galaxy(filename,outfilename,sfid_all,merger_bool_all):
    print(filename,outfilename)

    hdu = pyfits.open(filename)[0]

    array = pyfits.open(filename)[0].data
    new_array = congrid.congrid(array[61:162,61:162],(64,64))
    
    sfid = float( hdu.header['SUBH_ID'] )

    #mbi = np.where(sfid_all==sfid)[0]
    #mb = merger_bool_all[mbi]
    #print(mb,mbi,sfid_all)

    index = sfid_all==sfid

    mb = merger_bool_all[sfid_all==sfid]

    fu = pyfits.PrimaryHDU(new_array)

    fu.header['SUBH_ID']=sfid
    fu.header['ISMERGER']=float(mb)

    ful = pyfits.HDUList([fu])

    ful.writeto(outfilename,overwrite=True)
    

    return



def convert_galaxies(sd='subdir_010'):
    image_base='/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/snapshot_068/'

    morphology_catalog_file = 'morphology_catalog_068.hdf5'

    with h5py.File(morphology_catalog_file,'r') as catfile:
        snap_properties = catfile['snapshot_068']
        
        #Now, an intrinsic quantity that we want to learn about
        #Is there a major merger within +/- 500Myr ?
        merger_bool= np.asarray( catfile['snapshot_068']['Mergers_About1Gyr'].value)
        sfid = np.asarray( catfile['snapshot_068']['SubfindID'].value )
        print(sfid,merger_bool)


    files_105 = np.sort(np.asarray(glob.glob(image_base+str(sd)+'/images_subhalo_*/*WFC3-F105W_SB00.fits')))
    files_125 = np.sort(np.asarray(glob.glob(image_base+str(sd)+'/images_subhalo_*/*WFC3-F125W_SB00.fits')))
    files_160 = np.sort(np.asarray(glob.glob(image_base+str(sd)+'/images_subhalo_*/*WFC3-F160W_SB00.fits')))


    print(files_125.shape)

    if not os.path.lexists(sd):
        os.mkdir(sd)

    for f in files_105:
        outfilename=sd+'/rebin_'+os.path.basename(f)
        convert_galaxy(f,outfilename,sfid,merger_bool)
 
    for f in files_125:
        outfilename=sd+'/rebin_'+os.path.basename(f)
        convert_galaxy(f,outfilename,sfid,merger_bool)
        
    for f in files_160:
        outfilename=sd+'/rebin_'+os.path.basename(f)
        convert_galaxy(f,outfilename,sfid,merger_bool)
               

    return



if __name__=="__main__":


    st = time.time()

    convert_galaxies(sd='subdir_000')
    
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
