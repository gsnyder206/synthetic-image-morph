import cProfile
import pstats
import math
import string
import sys
import struct
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as pyplot
import matplotlib.colors as pycolors
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
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
import warnings
import subprocess
import photutils
import astropy
import astropy.cosmology
import astropy.io.fits as pyfits
import astropy.units as u
from astropy.constants import G
from astropy.cosmology import WMAP7,z_at_value
from astropy.coordinates import SkyCoord
import copy
import medianstats_bootstrap as msbs
import illustris_python as ilpy
import h5py
from parse_illustris_morphs import *
from PyML import machinelearning as pyml
from PyML import convexhullclassifier as cvx
import gfs_sublink_utils as gsu

def match_progdesc(snap, sfid, snapkeys,basepath='/astro/snyder_lab2/Illustris/Illustris-1'):
    this_snap_int = np.int32(snap[-3:])
    snaplist = []
    for s in snapkeys:
        snaplist.append(s[-3:])

    snaps_int = np.int64(np.asarray(snaplist))

    sublink_id,tree_id,mstar,sfr = gsu.sublink_id_from_subhalo(basepath,this_snap_int,sfid)
    tree = gsu.load_full_tree(basepath,tree_id)
    mmpb_sublink_id,snaps,mass,mstar,r1,r2 = gsu.mmpb_from_tree(tree,sublink_id)

    matched_mmpb = np.zeros_like(snaps_int)

    for i,snap in enumerate(snaps_int):
        tsi = np.where(snaps==snap)[0]
        if tsi.shape[0] == 1:
            matched_mmpb[i]=mmpb_sublink_id[tsi]
        else:
            matched_mmpb[i] = -1

    return matched_mmpb, sublink_id


if __name__=="__main__":

    bp = '/astro/snyder_lab2/Illustris/Illustris-1'  #simulation base path for merger trees

    catalogfile = "/astro/snyder_lab2/Illustris/MorphologyAnalysis/nonparmorphs_SB25_12filters_morestats.hdf5"   #morphology catalog file


    morphcat = h5py.File(catalogfile)
    topkey = morphcat.keys()[0]
    snapkeys = morphcat[topkey].keys()
    for snapkey in snapkeys:
        print snapkey
        print morphcat[topkey][snapkey].keys()
        subfind_id = morphcat[topkey][snapkey]['SubfindID'].value
        print subfind_id.shape
        for sfid in subfind_id[0:10]:
            id_grid,sublink_id = match_progdesc(snapkey,sfid,snapkeys)
            print id_grid

    
    z1key = 'snapshot_085'
    catalogkeys = morphcat[topkey][z1key].keys()
    stellar_mass = morphcat[topkey][z1key]['Mstar_Msun'].value
    sfr = morphcat[topkey][z1key]['SFR_Msunperyr'].value
    filters = morphcat[topkey][z1key]['Filters'].value
    print filters

    morphkeys = morphcat[topkey][z1key]['WFC3-F160W']['CAMERA1'].keys()
    print morphkeys


    halflightradius_Hband_z1_cam1 = morphcat[topkey][z1key]['WFC3-F160W']['CAMERA1']['RHALF'].value
