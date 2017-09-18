#import ez_galaxy
import math
import string
import sys
import struct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
import matplotlib.colors as pycolors
import matplotlib.cm as cm
#import utils
import numpy as np
#import array
#import astLib.astStats as astStats
import cPickle
import asciitable
import scipy.ndimage
import scipy.stats as ss
import scipy.signal
import scipy as sp
import scipy.odr as odr
import pyfits
import glob
import os
import make_color_image
import make_fake_wht
import gzip
import tarfile
import shutil
#import psfmatch
#from pygoods import *
#from sextutils import *
import cosmocalc
#from parse_hydroART_morphs import *
import numpy.ma as ma
import gfs_mockimage_code as gmc
#from sklearn import linear_model
import medianstats_bootstrap as msbs
import astropy
import astropy.io.fits as pyfits
import astropy.units as u
from astropy.cosmology import WMAP7,z_at_value

ilh = 0.704
illcos = astropy.cosmology.FlatLambdaCDM(H0=70.4,Om0=0.2726,Ob0=0.0456)


def f160_psf_fwhm(redshift):
    f160_fwhm = 0.199
    pskpc = illcos.kpc_proper_per_arcmin(redshift).value/60.0 #np.asarray( [cosmocalc.cosmocalc(z,H0=70.4,WM=0.2726,WV=0.7274)['PS_kpc'] for z in redshift])
    return f160_fwhm*pskpc

def f606_psf_fwhm(redshift):
    f606_fwhm = 0.115
    pskpc = illcos.kpc_proper_per_arcmin(redshift).value/60.0 #np.asarray( [cosmocalc.cosmocalc(z,H0=70.4,WM=0.2726,WV=0.7274)['PS_kpc'] for z in redshift])
    return f606_fwhm*pskpc

def f850_psf_fwhm(redshift):
    f850_fwhm = 0.110
    pskpc = illcos.kpc_proper_per_arcmin(redshift).value/60.0 #np.asarray( [cosmocalc.cosmocalc(z,H0=70.4,WM=0.2726,WV=0.7274)['PS_kpc'] for z in redshift])
    return f850_fwhm*pskpc


def wfc3_pix(redshift):
    wfc3_pix_opt = 0.060
    pskpc = illcos.kpc_proper_per_arcmin(redshift).value/60.0 #np.asarray( [cosmocalc.cosmocalc(z,H0=70.4,WM=0.2726,WV=0.7274)['PS_kpc'] for z in redshift])    
    return pskpc*wfc3_pix_opt

def nircam_pix(redshift):
    nircam_pix_opt = 0.032
    pskpc = illcos.kpc_proper_per_arcmin(redshift).value/60.0 #np.asarray( [cosmocalc.cosmocalc(z,H0=70.4,WM=0.2726,WV=0.7274)['PS_kpc'] for z in redshift])    
    return pskpc*wfc3_pix_opt

def f200_psf_fwhm(redshift):
    f200_fwhm = 0.0580
    pskpc = illcos.kpc_proper_per_arcmin(redshift).value/60.0 #np.asarray( [cosmocalc.cosmocalc(z,H0=70.4,WM=0.2726,WV=0.7274)['PS_kpc'] for z in redshift])    
    return pskpc*f200_fwhm

def f444_psf_fwhm(redshift):
    f444_fwhm = 0.13
    pskpc = illcos.kpc_proper_per_arcmin(redshift).value/60.0 #np.asarray( [cosmocalc.cosmocalc(z,H0=70.4,WM=0.2726,WV=0.7274)['PS_kpc'] for z in redshift])    
    return pskpc*f444_fwhm


def maxpixelsize(redshift):
    ztrans = 1.0
    alpha = 1.0
    pixsize = 0.710

    f606_target = f606_psf_fwhm(redshift)/2.0
    f200_target = f200_psf_fwhm(redshift)/2.0

    lowz = pixsize*np.ones_like(redshift)
    highz = pixsize*((1.0+ztrans)**alpha)*np.ones_like(redshift)/(1.0+redshift)**alpha

    #max = np.where(redshift <= ztrans, lowz, highz)
    #max = np.where(max < f200_target,max,f200_target)
    max = f200_target

    return max


def minpixelsize(redshift,Npixels=512.0):
    ztrans = 1.0
    alpha = 1.0
    fov = 75.0
    hardmin = 30.0

    pixsize = 0.40

    lowz = fov*np.ones_like(redshift)
    highz = fov*((1.0+ztrans)**alpha)*np.ones_like(redshift)/(1.0+redshift)**alpha

    min = np.where(redshift <= ztrans, lowz, highz)/Npixels
    min = np.where(min < hardmin/Npixels, (hardmin/Npixels)*np.ones_like(redshift), min)

    return min


def fov(redshift):
    ztrans = 1.0
    alpha = 1.0
    fovsize = 120.0

    lowz = fovsize*np.ones_like(redshift)
    highz = fovsize*((1.0+ztrans)**alpha)*np.ones_like(redshift)/(1.0+redshift)**alpha

    fov_values = np.where(redshift <= ztrans, lowz, highz)
    return fov_values


def illustris_softening(redshift):

    lowz = np.ones_like(redshift)*0.710
    highz = np.ones_like(redshift)*1.420
    soft = np.where(redshift <= 1.0, lowz, highz/(1.0+redshift) )

    return soft


if __name__=="__main__":
    zgrid = np.flipud(np.arange(0.2,12.0,0.01))
    tgrid = illcos.age(zgrid).value #np.asarray([cosmocalc.cosmocalc(z,H0=70.4,WM=0.2726,WV=0.7274)['zage_Gyr'] for z in zgrid])
    
    snaps = ['035','038','041','045','049','054','060','064','068','075','085','103']
    zsnaps = np.asarray([9.0023399,8.0121729,7.0054170,6.01075739886,4.99593346816,4.0079451114,3.00813107163,2.44422570456,2.00202813925,1.47197484527,0.997294225782,0.503047523246])

    zsnaps_time = np.interp(zsnaps,np.flipud(zgrid),np.flipud(tgrid))

    i1soft_kpc = 2.0*illustris_softening(zgrid)
    wfc3_pix_kpc = wfc3_pix(zgrid)
    wfc3_f160_fwhm_kpc = f160_psf_fwhm(zgrid)

    acs_f606_fwhm_kpc = f606_psf_fwhm(zgrid)
    acs_f850_fwhm_kpc = f850_psf_fwhm(zgrid)

    prop_max_pix_kpc = maxpixelsize(zgrid)
    prop_min_pix_kpc = minpixelsize(zgrid)

    jwst_f200_fwhm_kpc = f200_psf_fwhm(zgrid)
    jwst_f444_fwhm_kpc = f444_psf_fwhm(zgrid)

    fovs_dir = '/Users/gsnyder/Documents/Projects/Illustris_Morphology/Illustris-CANDELS/fovs'

    fovs_z05 = os.path.join(fovs_dir,'fov_outputs_103.txt')
    fovcols_z05 = asciitable.read(fovs_z05)
    fov_z05_kpc = fovcols_z05['col2']/(1.5)
    pix_z05_kpc = fov_z05_kpc/256.0
    medpix_z05 = np.median(pix_z05_kpc)
    madpix_z05 = msbs.MAD(pix_z05_kpc)
    z05_time = np.interp(0.5,np.flipud(zgrid),np.flipud(tgrid))


    fovs_z1 = os.path.join(fovs_dir,'fov_outputs_085.txt')
    fovcols_z1 = asciitable.read(fovs_z1)
    fov_z1_kpc = fovcols_z1['col2']/(2.0)
    pix_z1_kpc = fov_z1_kpc/256.0
    medpix_z1 = np.median(pix_z1_kpc)
    madpix_z1 = msbs.MAD(pix_z1_kpc)
    z1_time = np.interp(1.0,np.flipud(zgrid),np.flipud(tgrid))

    fovs_z2 = os.path.join(fovs_dir,'fov_outputs_068.txt')
    fovcols_z2 = asciitable.read(fovs_z2)
    fov_z2_kpc = fovcols_z2['col2']/(3.0)
    print np.min(fov_z2_kpc), np.max(fov_z2_kpc), np.median(fov_z2_kpc)
    pix_z2_kpc = fov_z2_kpc/256.0
    medpix_z2 = np.median(pix_z2_kpc)
    madpix_z2 = msbs.MAD(pix_z2_kpc)
    z2_time = np.interp(2.0,np.flipud(zgrid),np.flipud(tgrid))

    fovs_z3 = os.path.join(fovs_dir,'fov_outputs_060.txt')
    fovcols_z3 = asciitable.read(fovs_z3)
    fov_z3_kpc = fovcols_z3['col2']/(4.0)
    pix_z3_kpc = fov_z3_kpc/256.0
    medpix_z3 = np.median(pix_z3_kpc)
    madpix_z3 = msbs.MAD(pix_z3_kpc)
    z3_time = np.interp(3.0,np.flipud(zgrid),np.flipud(tgrid))


    zsnaps_soft = np.interp(zsnaps,np.flipud(zgrid),np.flipud(i1soft_kpc))

    lowx = 0.45 ; hix = 9.3
    f2 = pyplot.figure(figsize=(4.5,4.0), dpi=150)
    pyplot.subplots_adjust(left=0.11, right=0.99, bottom=0.11, top=0.89,wspace=0.0,hspace=0.0)
    vecbins = 14

    axi = f2.add_subplot(1,1,1)
    axi.locator_params(nbins=5,prune='both')    
    axi.set_xlabel('time [Gyr]',labelpad=1)
    axi.set_ylabel('size [kpc]',labelpad=1)
    axi.set_xlim(lowx,hix)
    axi.set_ylim(0.08,2.3)

    #axi.plot()

    i1 = axi.semilogy(tgrid,i1soft_kpc,marker='None',color='black',linewidth=1.0)
    snappoints = axi.plot(zsnaps_time,zsnaps_soft,linestyle='None',marker='*',color='LightGreen',markersize=8)
    #wfcpix = axi.semilogy(tgrid,wfc3_pix_kpc,marker='None',color='red',linewidth=1.0,linestyle='dotted')  #"WFC3-IR mosaic pixel (0.06'')",
    f160 = axi.semilogy(tgrid,wfc3_f160_fwhm_kpc,marker='None',color='red',linewidth=1.0,linestyle='dashed')
    f606 = axi.semilogy(tgrid,acs_f606_fwhm_kpc,marker='None',color='blue',linewidth=1.0,linestyle='dashed')
    #f606_half = axi.semilogy(tgrid,0.5*acs_f606_fwhm_kpc,marker='None',color='gray',linewidth=1.0,linestyle='solid')
    f200 = axi.semilogy(tgrid,jwst_f200_fwhm_kpc,marker='None',color='orange',linewidth=1.0,linestyle='dashed')
    prop_pix = axi.semilogy(tgrid,prop_max_pix_kpc,marker='None',color='SlateGray',linewidth=1.0,linestyle='dotted')
    #propmin_pix = axi.semilogy(tgrid,prop_min_pix_kpc,marker='None',color='gray',linewidth=1.0,linestyle='solid')

    #axi.errorbar([z1_time],[medpix_z1/2],madpix_z1/2,color='yellow')
    #axi.errorbar([z2_time],[medpix_z2/2],madpix_z2/2,color='yellow')
    #axi.errorbar([z3_time],[medpix_z3/2],madpix_z3/2,color='yellow')
    #axi.errorbar([z05_time],[medpix_z05/2],madpix_z05/2,color='yellow')


    axi.legend(('2x min. softening','"Illustris-CANDELS"','F160W FWHM','F606W FWHM','NIRCAM/F200W FWHM','image pixel'),loc='lower right',prop={'size':9},frameon=False,handlelength=4,numpoints=1)

    ticks=[0.1,0.2,0.5,1.0,2.0]
    zticks = [0.5,1,2,3,4,6]
    z_tticks = np.interp(zticks,np.flipud(zgrid),np.flipud(tgrid))

    majorlocator = matplotlib.ticker.FixedLocator(ticks)
    majorformatter = matplotlib.ticker.FixedFormatter(np.array(ticks,dtype='string'))
    #majorformatter.set_scientific(False)
    axi.yaxis.set_major_formatter(majorformatter) ; axi.yaxis.set_major_locator(majorlocator)
    axi.tick_params(axis='y',which='minor',left='off',right='off')

    topaxi = axi.twiny()
    topaxi.set_xlim(lowx,hix)
    toplocator = matplotlib.ticker.FixedLocator(z_tticks)
    topformatter = matplotlib.ticker.FixedFormatter(np.array(zticks,dtype='string'))
    topaxi.xaxis.set_major_locator(toplocator)
    topaxi.xaxis.set_major_formatter(topformatter)

    topaxi.set_xlabel('redshift')

    f2.savefig('Illustris_Resolution_Plot.pdf',format='pdf')
    pyplot.close(f2)







    lowx = 0.495 ; hix = 9.3
    f2 = pyplot.figure(figsize=(6.5,4.0), dpi=150)
    pyplot.subplots_adjust(left=0.09, right=0.99, bottom=0.11, top=0.89,wspace=0.0,hspace=0.0)
    vecbins = 14

    s=15
    ymax = 4.0
    axi = f2.add_subplot(1,1,1)
    axi.locator_params(nbins=5,prune='both')    
    axi.set_xlabel('time [Gyr]',labelpad=1,size=s)
    axi.set_ylabel('size [kpc]',labelpad=1,size=s)
    axi.set_xlim(lowx,hix)
    axi.set_ylim(0.2,ymax)

    #axi.plot()

    i1 = axi.semilogy(tgrid,i1soft_kpc,marker='None',color='black',linewidth=3.0)
    snappoints = axi.plot(zsnaps_time,zsnaps_soft,linestyle='None',marker='*',color='LightGreen',markersize=15)

    axi.annotate('WFC3',(3.0,2.0),xycoords='data',ha='center',va='center',size=s+3,color='Pink')
    axi.annotate('ACS',(7.0,3.5),xycoords='data',ha='center',va='center',size=s+3,color='White')

    xz = np.where(zgrid <= 3.0)[0]
    axi.fill_between(tgrid[xz],wfc3_f160_fwhm_kpc[xz],np.ones_like(xz)+ymax,interpolate=True,color='Blue',alpha=0.6)
    xz = np.where(zgrid <= 1.2)[0]
    axi.fill_between(tgrid[xz],acs_f850_fwhm_kpc[xz],np.ones_like(xz)+ymax,interpolate=True,color='Black',alpha=0.6)



    axi.legend(('2x min. softening','HST AR#13887','F160W FWHM','F606W FWHM','NIRCAM/F200W FWHM','image pixel'),loc='lower right',prop={'size':s},frameon=False,handlelength=4,numpoints=1)

    ticks=[0.1,0.2,0.5,1.0,2.0]
    zticks = [0.5,1,2,3,4,6,9]
    z_tticks = np.interp(zticks,np.flipud(zgrid),np.flipud(tgrid))

    majorlocator = matplotlib.ticker.FixedLocator(ticks)
    majorformatter = matplotlib.ticker.FixedFormatter(np.array(ticks,dtype='string'))
    #majorformatter.set_scientific(False)
    axi.yaxis.set_major_formatter(majorformatter) ; axi.yaxis.set_major_locator(majorlocator)
    axi.tick_params(axis='y',which='minor',left='off',right='off',labelsize=s)

    topaxi = axi.twiny()
    topaxi.set_xlim(lowx,hix)
    toplocator = matplotlib.ticker.FixedLocator(z_tticks)
    topformatter = matplotlib.ticker.FixedFormatter(np.array(zticks,dtype='string'))
    topaxi.xaxis.set_major_locator(toplocator)
    topaxi.xaxis.set_major_formatter(topformatter)

    topaxi.set_xlabel('redshift',size=s,labelpad=1)




    f2.savefig('Illustris_Scales_Plot_HST.pdf',format='pdf')

    axi.cla()
    axi.locator_params(nbins=5,prune='both')    
    axi.set_xlabel('time [Gyr]',labelpad=1,size=s)
    axi.set_ylabel('size [kpc]',labelpad=1,size=s)
    axi.set_xlim(lowx,hix)
    axi.set_ylim(0.2,ymax)

    i1 = axi.semilogy(tgrid,i1soft_kpc,marker='None',color='black',linewidth=3.0)
    snappoints = axi.plot(zsnaps_time,zsnaps_soft,linestyle='None',marker='*',color='LightGreen',markersize=15)


    axi.annotate('WFC3',(3.0,2.0),xycoords='data',ha='center',va='center',size=s+3,color='Pink')
    axi.annotate('ACS',(7.0,3.5),xycoords='data',ha='center',va='center',size=s+3,color='White')
    axi.annotate('NC-short',(5.0,0.6),xycoords='data',ha='center',va='center',size=s+3,color='Black')
    axi.annotate('NC-long',(1.3,3.4),xycoords='data',ha='center',va='center',size=s+3,color='Yellow')


    xz = np.where(zgrid <= 4.0)[0]
    axi.fill_between(tgrid[xz],jwst_f200_fwhm_kpc[xz],np.ones_like(xz)+ymax,interpolate=True,color='Orange',alpha=0.6)
    xz = np.where(zgrid <= hix)[0]
    axi.fill_between(tgrid[xz],jwst_f444_fwhm_kpc[xz],np.ones_like(xz)+ymax,interpolate=True,color='Red',alpha=0.6)

    xz = np.where(zgrid <= 3.0)[0]
    axi.fill_between(tgrid[xz],wfc3_f160_fwhm_kpc[xz],np.ones_like(xz)+ymax,interpolate=True,color='Blue',alpha=0.6)
    xz = np.where(zgrid <= 1.2)[0]
    axi.fill_between(tgrid[xz],acs_f850_fwhm_kpc[xz],np.ones_like(xz)+ymax,interpolate=True,color='Black',alpha=0.6)

    axi.legend(('2x min. softening','HST AR#13887','F160W FWHM','F606W FWHM','NIRCAM/F200W FWHM','image pixel'),loc='lower right',prop={'size':s},frameon=False,handlelength=4,numpoints=1)
    majorlocator = matplotlib.ticker.FixedLocator(ticks)
    majorformatter = matplotlib.ticker.FixedFormatter(np.array(ticks,dtype='string'))
    #majorformatter.set_scientific(False)
    axi.yaxis.set_major_formatter(majorformatter) ; axi.yaxis.set_major_locator(majorlocator)
    axi.tick_params(axis='y',which='minor',left='off',right='off',labelsize=s)


    f2.savefig('Illustris_Scales_Plot_JWST.pdf',format='pdf')
    pyplot.close(f2)



    f2 = pyplot.figure(figsize=(6.5,6.5), dpi=150)
    pyplot.subplots_adjust(left=0.10, right=0.99, bottom=0.08, top=0.99,wspace=0.0,hspace=0.0)

    Npix = 512

    axi = f2.add_subplot(4,1,1)
    axi.locator_params(nbins=5,prune='both')    
    n,x,x = axi.hist(fov_z3_kpc,bins=100,range=[0.0,500.0])
    axi.set_ylabel('number')
    axi.set_xticklabels([])
    axi.annotate('z=3',(0.5,0.80),xycoords='axes fraction',size=9,color='black')
    #axi.plot([Npix*maxpixelsize(np.asarray([3.0])),Npix*maxpixelsize(np.asarray([3.0]))],[0,np.max(n)*1.2],color='red')
    #axi.plot([Npix*minpixelsize(np.asarray([3.0])),Npix*minpixelsize(np.asarray([3.0]))],[0,np.max(n)*1.2],color='red')
    axi.plot([fov(np.asarray([3.0])),fov(np.asarray([3.0]))],[0,np.max(n)*1.2],color='red')

    axi = f2.add_subplot(4,1,2)
    axi.locator_params(nbins=5,prune='both')    
    n,x,x = axi.hist(fov_z2_kpc,bins=100,range=[0.0,500.0])
    axi.set_xticklabels([])
    axi.annotate('z=2',(0.5,0.80),xycoords='axes fraction',size=9,color='black')
    #axi.plot([Npix*maxpixelsize(np.asarray([2.0])),Npix*maxpixelsize(np.asarray([2.0]))],[0,np.max(n)*1.2],color='red')
    #axi.plot([Npix*minpixelsize(np.asarray([2.0])),Npix*minpixelsize(np.asarray([2.0]))],[0,np.max(n)*1.2],color='red')
    axi.plot([fov(np.asarray([2.0])),fov(np.asarray([2.0]))],[0,np.max(n)*1.2],color='red')

    axi = f2.add_subplot(4,1,3)
    axi.locator_params(nbins=5,prune='both')    
    n,x,x = axi.hist(fov_z1_kpc,bins=100,range=[0.0,500.0])
    axi.set_xticklabels([])
    axi.annotate('z=1',(0.5,0.80),xycoords='axes fraction',size=9,color='black')
    #axi.plot([Npix*maxpixelsize(np.asarray([1.0])),Npix*maxpixelsize(np.asarray([1.0]))],[0,np.max(n)*1.2],color='red')
    #axi.plot([Npix*minpixelsize(np.asarray([1.0])),Npix*minpixelsize(np.asarray([1.0]))],[0,np.max(n)*1.2],color='red')
    axi.plot([fov(np.asarray([1.0])),fov(np.asarray([1.0]))],[0,np.max(n)*1.2],color='red')

    axi = f2.add_subplot(4,1,4)
    axi.locator_params(nbins=5,prune='both')    
    n,x,x = axi.hist(fov_z05_kpc,bins=100,range=[0.0,500.0])
    axi.annotate('z=0.5',(0.5,0.80),xycoords='axes fraction',size=9,color='black')
    #axi.plot([Npix*maxpixelsize(np.asarray([0.5])),Npix*maxpixelsize(np.asarray([0.5]))],[0,np.max(n)*1.2],color='red')
    #axi.plot([Npix*minpixelsize(np.asarray([0.5])),Npix*minpixelsize(np.asarray([0.5]))],[0,np.max(n)*1.2],color='red')
    axi.plot([fov(np.asarray([0.5])),fov(np.asarray([0.5]))],[0,np.max(n)*1.2],color='red')

    axi.set_xlabel('fov [kpc]')


    f2.savefig('Illustris_FOV_Plot.pdf',format='pdf')
    pyplot.close(f2)

    import cosmocalc
    #the target FOV, in physical kpc
    fov_set = 120.0

    #snapshot information, with exact redshifts, from broadband*.config files

    #alternative semi-comoving fov target value
    ztrans = 1.0
    alpha = 1.0
    lowz = fov_set*np.ones_like(zsnaps)
    highz = fov_set*((1.0+ztrans)**alpha)*np.ones_like(zsnaps)/(1.0+zsnaps)**alpha

    fov_fudge = np.where(zsnaps <= ztrans, lowz, highz)


    #exact angular size of a pixel at exact z above
    arcsec_pix_exact = 0.029000

    for i,z in enumerate(zsnaps):

        #PSkpc is the number of physical kpc per arcsecond at redshift z
        PSkpc = cosmocalc.cosmocalc(z,H0=70.4,WM=0.2726,WV=0.7274)['PS_kpc']
        #I've used the Illustris cosmology values here.. I think

        #convert pixel size in arcsec to physical kpc
        pixsize = arcsec_pix_exact*PSkpc  #maxpixelsize(np.asarray([z]))

        #decide which fov to target-- fixed in physical kpc or semi-comoving version
        fov_use = fov_set
        #fov_use = fov_fudge[i]

        #the float number of pixels required to give exact FOV and pixel size
        Npix_flt = fov_use/pixsize
        
        #round down to nearest integer value
        Npix_int = int(Npix_flt)

        #compute the precise new FOV value required to give the correct pixel size with Npix_int
        fov_new = Npix_int*pixsize

        print 'snapshot_{:3s} z={:10.8f}, pix={:8.6f} kpc = {:6.4f} arcsec, N={:3.0f} pixels, FOV={:8.6f} pkpc'.format(snaps[i], z, pixsize, arcsec_pix_exact, Npix_int, fov_new)


