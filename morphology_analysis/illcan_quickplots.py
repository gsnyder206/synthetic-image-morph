import math
import string
import sys
import struct
import matplotlib
#matplotlib.use('PDF')
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
from scipy.integrate import quad
import glob
import os
import gzip
import shutil
import congrid
import astropy.io.ascii as ascii
import warnings
import subprocess
import astropy
import astropy.io.fits as pyfits
import astropy.units as u
from astropy.cosmology import WMAP7,z_at_value
import copy
import datetime
import ezgal
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.collections import PolyCollection
import parse_illustris_morphs as pim
import illustris_python as ilpy
from parse_illustris_morphs import *
from PyML import machinelearning as pyml
from PyML import convexhullclassifier as cvx
import statmorph



ilh = 0.704
illcos = astropy.cosmology.FlatLambdaCDM(H0=70.4,Om0=0.2726,Ob0=0.0456)
sq_arcsec_per_sr = 42545170296.0

def make_pc_dict(mo,fi):

    parameters = ['C','M20','GINI','ASYM','MPRIME','I','D']
    pcd = {}
    pcd['C'] = mo.cc[:,:,fi,0].flatten()
    pcd['M20'] = mo.m20[:,:,fi,0].flatten()
    pcd['GINI'] = mo.gini[:,:,fi,0].flatten()
    pcd['ASYM'] = mo.asym[:,:,fi,0].flatten()
    pcd['MPRIME'] = mo.mid1_mstat[:,:,fi,0].flatten()
    pcd['I'] = mo.mid1_istat[:,:,fi,0].flatten()
    pcd['D'] = mo.mid1_dstat[:,:,fi,0].flatten()

    npmorph = pyml.dataMatrix(pcd,parameters)
    pc = pyml.pcV(npmorph)

    return parameters, pcd, pc, pcd


def pc1_sizemass_panel(f1,nr,nc,nt,size,mass,pc1,xlim=[5.0e9,8.0e11],ylim={'sizemass':[0.7,20.0]},gridsize=12,vlim=[-2,2]):

    rlim = np.log10(np.asarray(ylim['sizemass']))
    mlim = np.log10(np.asarray(xlim))
    extent=[mlim[0],mlim[1],rlim[0],rlim[1]]
    print extent
    s=11

    nzi = np.where(size > 0.0)[0]

    axi = f1.add_subplot(nr,nc,nt)
    axi.locator_params(nbins=5,prune='both')
    print mass.shape, size.shape, pc1.shape

    axi.hexbin(mass[nzi],size[nzi],C=-1.0*pc1[nzi],gridsize=gridsize,xscale='log',yscale='log',reduce_C_function=np.median,mincnt=3,extent=extent,vmin=vlim[0],vmax=vlim[1])


    axi.tick_params(axis='both',which='major',labelsize=s)
    axi.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    if nt % nc == 1:
        axi.set_ylabel(r'$r_{1/2}$ [kpc]',size=s,labelpad=1)
    else:
        axi.set_yticklabels([])

    if nt > (nr-1)*nc:
        axi.set_xlabel(r'$M_{*}$ [$M_{\odot}$]',size=s,labelpad=1)
    else:
        axi.set_xticklabels([])

    return axi,s


def pc1_sfrmass_panel(f1,nr,nc,nt,sfr,mass,pc1,xlim=[5.0e9,8.0e11],ylim={'sfrmass':[5,500.0]},gridsize=12,vlim=[-2,2]):

    rlim = np.log10(np.asarray(ylim['sfrmass']))
    mlim = np.log10(np.asarray(xlim))
    extent=[mlim[0],mlim[1],rlim[0],rlim[1]]
    print extent
    s=11

    nzi = np.where(sfr > 0.0)[0]

    axi = f1.add_subplot(nr,nc,nt)
    axi.locator_params(nbins=5,prune='both')
    print mass.shape, sfr.shape, pc1.shape

    axi.hexbin(mass[nzi],sfr[nzi],C=-1.0*pc1[nzi],gridsize=gridsize,xscale='log',yscale='log',reduce_C_function=np.median,mincnt=3,extent=extent,vmin=vlim[0],vmax=vlim[1])


    axi.tick_params(axis='both',which='major',labelsize=s)

    if nt % nc == 1:
        axi.set_ylabel(r'SFR [$M_{\odot}\ yr^{-1}$]',size=s,labelpad=1)
    else:
        axi.set_yticklabels([])

    if nt > (nr-1)*nc:
        axi.set_xlabel(r'$M_{*}$ [$M_{\odot}$]',size=s,labelpad=1)
    else:
        axi.set_xticklabels([])

    return axi,s



def do_pc1_sizemass(figfile,data=None, snaps=None,filters=None,**kwargs):
    assert data is not None
    assert snaps is not None


    f1 = pyplot.figure(figsize=(6.5,3.5), dpi=150)
    pyplot.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.98,wspace=0.0,hspace=0.05)

    for i,s,f in zip(range(len(snaps)),snaps,filters):
        mo = data['morph'][s]
        filternames = mo.filters
        fi = np.where(filternames==f)[0]
        assert fi.shape[0]==1
        fi=fi[0]

        print filternames[fi]

        redshift = mo.redshift[0,0,fi,0]

        print redshift

        kpc_per_arcsec = illcos.kpc_proper_per_arcmin(redshift).value/60.0

        size_pixels = data['morph'][s].rhalfc[:,:,fi,0].flatten() #note:rhalfe values incorrect in first parsing
        pix_arcsec = data['morph'][s].pix_arcsec[0,0,fi,0].flatten()

        print size_pixels.shape
        print pix_arcsec.shape
        print fi.shape
        print kpc_per_arcsec

        size_kpc = size_pixels*pix_arcsec*kpc_per_arcsec
        print size_kpc.shape
        print pix_arcsec

        print np.max(size_kpc), np.max(size_pixels)

        mstar_1 = data['subhalos'][s]['SubhaloMassInRadType'][:,4].flatten()*(1.0e10)/ilh
        print mstar_1.shape
        mass = np.swapaxes(np.tile(mstar_1,(4,1)),0,1)
        print mass.shape

        parameters,npdict,pc,pcd = make_pc_dict(mo,fi)
        pc1 = pc.X[:,0].flatten()
        print pc1.shape

        axi,lsize = pc1_sizemass_panel(f1,2,3,i+1,size_kpc,mass.flatten(),pc1,**kwargs)
        axi.annotate('z={:4.1f}'.format(redshift),(0.85,0.90),xycoords='axes fraction',size=lsize,color='black',ha='center',va='center')
        axi.annotate(f,(0.75,0.10),xycoords='axes fraction',size=lsize,color='black',ha='center',va='center')

    f1.savefig(figfile)
    pyplot.close(f1)
    return

    


def do_pc1_sfrmass(figfile,data=None, snaps=None,filters=None,**kwargs):
    assert data is not None
    assert snaps is not None


    f1 = pyplot.figure(figsize=(6.5,3.5), dpi=150)
    pyplot.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.98,wspace=0.0,hspace=0.05)

    for i,s,f in zip(range(len(snaps)),snaps,filters):
        mo = data['morph'][s]
        filternames = mo.filters
        fi = np.where(filternames==f)[0]
        assert fi.shape[0]==1
        fi=fi[0]

        print filternames[fi]

        redshift = mo.redshift[0,0,fi,0]

        print redshift

        kpc_per_arcsec = illcos.kpc_proper_per_arcmin(redshift).value/60.0

        size_pixels = data['morph'][s].rhalfc[:,:,fi,0].flatten() #note:rhalfe values incorrect in first parsing
        pix_arcsec = data['morph'][s].pix_arcsec[0,0,fi,0].flatten()

        print size_pixels.shape
        print pix_arcsec.shape
        print fi.shape
        print kpc_per_arcsec

        size_kpc = size_pixels*pix_arcsec*kpc_per_arcsec
        print size_kpc.shape
        print pix_arcsec

        print np.max(size_kpc), np.max(size_pixels)

        mstar_1 = data['subhalos'][s]['SubhaloMassInRadType'][:,4].flatten()*(1.0e10)/ilh
        print mstar_1.shape
        mass = np.swapaxes(np.tile(mstar_1,(4,1)),0,1)
        print mass.shape

        sfr_1 = data['subhalos'][s]['SubhaloSFR'][:].flatten()
        sfr = np.swapaxes(np.tile(sfr_1,(4,1)),0,1)

        parameters,npdict,pc,pcd = make_pc_dict(mo,fi)
        pc1 = pc.X[:,0].flatten()
        print pc1.shape

        axi,lsize = pc1_sfrmass_panel(f1,2,3,i+1,sfr.flatten(),mass.flatten(),pc1,**kwargs)
        axi.annotate('z={:4.1f}'.format(redshift),(0.85,0.90),xycoords='axes fraction',size=lsize,color='black',ha='center',va='center')
        axi.annotate(f,(0.75,0.10),xycoords='axes fraction',size=lsize,color='black',ha='center',va='center')

    f1.savefig(figfile)
    pyplot.close(f1)
    return



def do_candels_loading(filename = 'MorphDataObjectLight_SB25.pickle',hstonly=False):

    snaps = ['snapshot_060','snapshot_064','snapshot_068','snapshot_075','snapshot_085','snapshot_103']
    if hstonly is True:
        filters =  ['WFC3-F160W','WFC3-F160W','WFC3-F160W','WFC3-F105W','WFC3-F105W','ACS-F814W']
    else:
        filters =  ['NC-F200W','NC-F150W','NC-F150W','NC-F115W','ACS-F814W','ACS-F606W']

    data_dict,snaps = pim.load_everything(snaps=snaps,filename=filename)

    return snaps, filters, data_dict


def do_jandels_loading(filename = 'MorphDataObjectLight_SB27.pickle'):

    snaps = ['snapshot_041','snapshot_045','snapshot_049','snapshot_054','snapshot_060','snapshot_064']
    filters =  ['NC-F356W','NC-F356W','NC-F277W','NC-F200W','NC-F200W','NC-F150W']

    data_dict,snaps = pim.load_everything(snaps=snaps,filename=filename)

    return snaps, filters, data_dict


def do_all_plots(snaps,fils,data,label='CANDELS',**kwargs):

    do_pc1_sizemass('PC1_sizemass_'+label+'.pdf',data=data,snaps=snaps,filters=fils,**kwargs)

    do_pc1_sfrmass('PC1_sfrmass_'+label+'.pdf',data=data,snaps=snaps,filters=fils,vlim=[-2,1],gridsize=12,**kwargs)


    return

if __name__=="__main__":

    #as of 4/1/2016:
    #available_filters = ['WFC3-F336W','ACS-F435W','ACS-F606W','ACS-F814W','WFC3-F105W','WFC3-F160W','NC-F115W','NC-F150W','NC-F200W','NC-F277W','NC-F356W','NC-F444W']
    snaps, filters, data_dict = do_candels_loading()

    do_pc1_sizemass('PC1_sizemass.pdf', data=data_dict, snaps=snaps, filters=filters)
