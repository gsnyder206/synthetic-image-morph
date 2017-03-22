import astropy
import astropy.io.fits as pyfits
import math
import matplotlib
import matplotlib.pyplot as pyplot
import numpy as np
import scipy.ndimage
import scipy as sp
import astropy.cosmology
import astropy.io.ascii as ascii
import pandas as pd

tngh=0.6774
tngcos=astropy.cosmology.FlatLambdaCDM(H0=67.74,Om0=0.3089,Ob0=0.0486)
tngcos.snaps=np.asarray([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
       51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
       68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
       85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99])

tngcos.scales=np.asarray([ 0.0476,  0.0625,  0.0769,  0.0833,  0.0909,  0.0964,  0.1   ,
        0.1058,  0.1111,  0.1161,  0.1216,  0.125 ,  0.1334,  0.1429,
        0.1464,  0.1533,  0.1606,  0.1667,  0.1763,  0.1847,  0.1935,
        0.2   ,  0.2124,  0.2226,  0.2332,  0.25  ,  0.2561,  0.2677,
        0.279 ,  0.2902,  0.3012,  0.3121,  0.3228,  0.3333,  0.3439,
        0.3543,  0.3645,  0.3747,  0.3848,  0.3948,  0.4   ,  0.4147,
        0.4246,  0.4344,  0.4441,  0.4538,  0.4635,  0.4731,  0.4827,
        0.4923,  0.5   ,  0.5115,  0.521 ,  0.5306,  0.5401,  0.5496,
        0.5591,  0.5687,  0.5782,  0.5882,  0.5973,  0.6069,  0.6164,
        0.626 ,  0.6357,  0.6453,  0.655 ,  0.6667,  0.6744,  0.6841,
        0.6939,  0.7037,  0.7143,  0.7235,  0.7334,  0.7434,  0.7534,
        0.7635,  0.7692,  0.7838,  0.794 ,  0.8043,  0.8146,  0.825 ,
        0.8333,  0.8459,  0.8564,  0.8671,  0.8778,  0.8885,  0.8993,
        0.9091,  0.9212,  0.9322,  0.9433,  0.9545,  0.9657,  0.9771,
        0.9885,  1.    ])

tngcos.redshifts= 1.0/tngcos.scales - 1.0

tngcos.fullsnap=np.asarray([False, False,  True,  True,  True, False,  True, False,  True,
       False, False,  True, False,  True, False, False, False,  True,
       False, False, False,  True, False, False, False,  True, False,
       False, False, False, False, False, False,  True, False, False,
       False, False, False, False,  True, False, False, False, False,
       False, False, False, False, False,  True, False, False, False,
       False, False, False, False, False,  True, False, False, False,
       False, False, False, False,  True, False, False, False, False,
        True, False, False, False, False, False,  True, False, False,
       False, False, False,  True, False, False, False, False, False,
       False,  True, False, False, False, False, False, False, False,  True], dtype=bool)

if __name__=="__main__":

    total_GB = 0.0

    for i,sn in enumerate(tngcos.snaps):
        z=tngcos.redshifts[i]

        if z <= 0.30:
            Ng=6500
        elif z <=1.0:
            Ng=5000
        elif z <= 2.0:
            Ng=3000
        elif z <= 3.0:
            Ng=1600
        elif z <= 4.0:
            Ng=600
        else:
            Ng=100
            


        if z < 0.3:
            imtype='LSST'
            pix_min_as=0.2   # ~LSST pixel
            res_as=0.7
            fov_target_kpc = 120.0
            pix_target_kpc = 0.700

        elif z < 1.0:
            imtype='ST'
            pix_min_as=0.1  #e.g., Euclid, WFIRST
            res_as=0.20
            fov_target_kpc = 120.0
            pix_target_kpc = 0.700
        else:
            imtype='ST'
            pix_min_as=0.032   #~Nircam-Short pixel
            res_as=0.10
            fov_target_kpc = 240.0/(1.0 + z)
            pix_target_kpc = 1.40/(1.0 + z)

        full_bool=tngcos.fullsnap[i]
        if full_bool == True:
            Ncam=10 #face-on, edge-on, 3 axes, 7 iso/random ?
            if imtype=='ST':
                filter_file='filters_rest'
                flist=pd.read_table(filter_file+'.txt',header=None)
                bbz=z*1.0
            else:
                filter_file='filters_rest'
                flist=pd.read_table(filter_file+'.txt',header=None)
                bbz=max([z*1.0,0.1])
        else:
            Ncam=5  #face-on, edge-on, 3 iso/random
            if imtype=='ST':
                filter_file='filters_rest'
                flist=pd.read_table(filter_file+'.txt',header=None)
                bbz=z*1.0
            else:
                filter_file='filters_rest'
                flist=pd.read_table(filter_file+'.txt',header=None)
                bbz=max([z*1.0,0.1])


        Nfils=flist[0].shape[0]

        #what else to decide?
        # broadband_redshift
        # add_redshift
        #FOV [kpc]
        #pixel_size_arcsec (add_redshift)

        scale_kpc_per_as = tngcos.kpc_proper_per_arcmin(bbz).value/60.0

        pix_target_as = max([pix_target_kpc/scale_kpc_per_as,pix_min_as])
        pix_target_kpc=scale_kpc_per_as*pix_target_as
        #pix_target_as = pix_target_kpc/scale_kpc_per_as

        res_kpc = scale_kpc_per_as*res_as


        rough_npix = fov_target_kpc/pix_target_kpc
        int_npix = int(rough_npix)+1
        
        fov_new = int_npix*pix_target_kpc


        Naux=3

        pix_per_galaxy=(int_npix**2)*(Nfils+Naux)*Ncam
        bytes_per_pix=4.0

        GB_per_galaxy=bytes_per_pix*pix_per_galaxy/1.0e9

        GB_estimated = Ng*GB_per_galaxy

        total_GB = total_GB + GB_estimated

        print('{:03d}  {:9.6f}   {:5s}   {:4s}   {:3d}    {:20s} '
              '{:5.3f}    {:5.3f}    {:5.3f}    {:7.3f}    {:4d}  {:4d}     {:5.2f}  {:7d}   {:7.1f}'.format(sn,z,str(full_bool),imtype, Ncam, filter_file,
                                                                                               pix_target_kpc,pix_target_as,res_kpc,fov_new,int_npix, Nfils, GB_per_galaxy,Ng,GB_estimated )  )

    print('Expected TB Size: ', total_GB/1.0e3)
