import math
import string
import sys
import struct
import matplotlib
matplotlib.use('TkAgg')
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
import warnings
import subprocess
import photutils
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import *
import astropy.io.fits as pyfits
import datetime
import medianstats_bootstrap as msbs
#want my custom skimage latest version not ssbx
sys.path = ['/Users/gsnyder/ssbvirt/ssbx-osx/lib/python2.7/site-packages']+sys.path
import skimage.transform
import copy
import region_grow

def arcsec_per_radian():
    return 3600.0*(180.0/math.pi)

def mad_value(array_input):
    return msbs.MAD(array_input)

def median_value(array_input):
    return np.median(array_input)

def mean_value(array_input):
    return np.mean(array_input)

def std_value(array_input):
    return np.std(array_input)

def var_value(array_input):
    return np.var(array_input)

def madvar_value(array_input):
    return msbs.MAD(array_input)**2

def central_moments(image, i_list, j_list, xc=None, yc=None):

    xi = np.float32(np.arange(image.shape[0]))+0.5
    pxpos,pypos = np.meshgrid(xi,xi)

    mu_00 = np.sum(image)

    if xc==None:
        moment_array = (pxpos)*image
        xc= np.sum(moment_array)/mu_00
    if yc==None:
        moment_array = (pypos)*image
        yc=np.sum(moment_array)/mu_00

    x_offsets = pxpos - xc
    y_offsets = pypos - yc
    #print xc, yc, mu_00, np.min(xi), np.max(xi)

    mu_ij = []
    eta_ij = []
    for index,i in enumerate(i_list):
        j=j_list[index]
        moment_array = ((x_offsets)**i)*((y_offsets)**j)*image
        mu_ij.append(np.sum(moment_array))
        eta_ij.append( mu_ij[index]/(mu_00**(1.0 + float(i+j)/2.0)))

    return np.asarray(mu_ij), np.asarray(eta_ij)



class galdata:
    def __init__(self):
        self.description='Lotz Morphology Input/Output object'
        self.hu_moments = [None]
        self.fs93_moments = [None]
        self.lotz_morph_status = 'Not Started'
        self.petro_segmap = None

    def run_lotz_morphs(self):
        #check for essential inputs
        correct_init = hasattr(self,'galaxy_image')
        correct_init = correct_init and hasattr(self,'galaxy_segmap')
        correct_init = correct_init and hasattr(self,'npix')
        correct_init = correct_init and hasattr(self,'pixel_xpos')
        correct_init = correct_init and hasattr(self,'pixel_ypos')
        correct_init = correct_init and hasattr(self,'xcentroid')
        correct_init = correct_init and hasattr(self,'ycentroid')
        correct_init = correct_init and hasattr(self,'elongation')
        correct_init = correct_init and hasattr(self,'pa_radians')
        correct_init = correct_init and hasattr(self,'skysig')
        correct_init = correct_init and hasattr(self,'pixelscale_arcsec')
        correct_init = correct_init and hasattr(self,'psf_fwhm_arcsec')
        correct_init = correct_init and hasattr(self,'magseg')
        correct_init = correct_init and hasattr(self,'magseg_err')
        correct_init = correct_init and hasattr(self,'rproj_arcsec')
        correct_init = correct_init and hasattr(self,'abzp')

        if correct_init:
            #carry out calculations
            self.rpetro_eta = 0.2
            self.extent_rpetro = 1.5
            self.lotz_morph_status = 'Initialized'


            #check S/N per galaxy pixel inside aperture defined by segmap
            self.snpix_init = self.sn_per_pixel(self.galaxy_image,self.galaxy_segmap)
            self.morph_hdu.header['SNP_INIT']=(round(self.snpix_init,8),'Initial average S/N per pixel')

            if self.snpix_init < 1.0:
                self.lotz_morph_status = 'Error: too faint!'
                print '        Exiting Morph calculation with status: '+self.lotz_morph_status
                print '                        Average S/N per pixel: {:7.3f}'.format(self.snpix_init)

                return
            else:
                self.lotz_morph_status = 'OK signal-to-noise (0)'


            #initial radius calculation, result in pixels
            self.rp_circ_1,self.rp_circ_status_1,self.rp_circ_err_1 = self.rpetro_circ(xcenter=self.xcentroid, ycenter=self.ycentroid)
            print '        Found circular Petrosian Radius (1):  {:8.4f}  {:8.4f}  {:25s}      {:8.4f}  {:8.4f} '.format(self.rp_circ_1, self.rp_circ_err_1, self.rp_circ_status_1, self.xcentroid, self.ycentroid)

            if self.rp_circ_status_1 != 'Positive R_pet':
                self.lotz_morph_status = 'Error: Poor Measurement of circular r_p (1)'
                print '        Exiting Morph calculation with status: '+self.lotz_morph_status
                return
            else:
                self.lotz_morph_status = 'Computed circular r_p (1)'



            #use fixed skybox to minimize computation inside minimized asymmetry function
            bkg_dif = (self.skybox - self.rot_skybox)
            self.a_bkg = np.sum(np.abs(bkg_dif))/np.sum(np.ones_like(self.skybox))



            #initial asymmetry minimization using rpc1
            self.asym1,self.xcen_a1,self.ycen_a1,self.asym1_message = self.compute_asym(xcenter=self.xcentroid,ycenter=self.ycentroid,extent=self.rp_circ_1*self.extent_rpetro,a_bkg=self.a_bkg)
            print '        Found Asymmetry & Center (1)       :  {:8.4f}  {:8.4f}  {:8.4f}    {:45s}'.format(self.asym1,self.xcen_a1,self.ycen_a1, self.asym1_message)
            if self.asym1==99.0:
                self.lotz_morph_status = 'Error: Poor Measurement of Asymmetry & Center (1)'
                print '        Exiting Morph calculation with status: '+self.lotz_morph_status+', message: '+self.asym1_message
                return
            else:
                self.lotz_morph_status = 'Computed Asymmetry & Center (1)'


            #recompute rpc2
            self.rp_circ_2,self.rp_circ_status_2,self.rp_circ_err_2 = self.rpetro_circ(xcenter=self.xcen_a1, ycenter=self.ycen_a1)
            print '        Found circular Petrosian Radius (2):  {:8.4f}  {:8.4f}  {:25s}  '.format(self.rp_circ_2, self.rp_circ_err_2, self.rp_circ_status_2)
            if self.rp_circ_status_2 != 'Positive R_pet':
                self.lotz_morph_status = 'Error: Poor Measurement of circular r_p (2)'
                print '        Exiting Morph calculation with status: '+self.lotz_morph_status
                return
            else:
                self.lotz_morph_status = 'Computed circular r_p (2)'

            #final asymmetry centering
            self.asym2,self.xcen_a2,self.ycen_a2,self.asym2_message = self.compute_asym(xcenter=self.xcen_a1,ycenter=self.ycen_a1,extent=self.rp_circ_2*self.extent_rpetro,a_bkg=self.a_bkg)
            
            asym2,self.ga2,self.ba2 = self.galaxy_asymmetry(np.asarray([self.xcen_a1,self.ycen_a1]),radius=self.rp_circ_2*self.extent_rpetro,a_bkg=self.a_bkg)

            print '        Found Asymmetry & Center (2)       :  {:8.4f}  {:8.4f}  {:8.4f}  {:8.4f}  {:8.4f}  {:8.4f}  {:8.4f} '.format(self.asym2,self.xcen_a2,self.ycen_a2, self.a_bkg, self.ga2, self.ba2, asym2)
            if self.asym2==99.0:
                self.lotz_morph_status = 'Error: Poor Measurement of Asymmetry & Center (2)'
                print '        Exiting Morph calculation with status: '+self.lotz_morph_status+', message: '+self.asym2_message
                return
            else:
                self.lotz_morph_status = 'Computed Asymmetry & Center (2)'

            #compute rpe, once and only once
            self.rp_ellip,self.rp_ellip_status,self.rp_ellip_err = self.rpetro_ellip(xcenter=self.xcen_a2, ycenter=self.ycen_a2)
            print '        Found elliptical Petrosian Radius  :  {:8.4f}  {:8.4f}  {:25s}  '.format(self.rp_ellip, self.rp_ellip_err, self.rp_ellip_status)
            if self.rp_ellip_status != 'Positive R_pet':
                self.lotz_morph_status = 'Error: Poor Measurement of elliptical r_p'
                print '        Exiting Morph calculation with status: '+self.lotz_morph_status
                return
            else:
                self.lotz_morph_status = 'Computed elliptical r_p'


            self.morph_hdu.header['RPC']=(round(self.rp_circ_2,8),'Petrosian Circular Radius (pixels)')
            self.morph_hdu.header['RPC_ERR']=(round(self.rp_circ_err_2,8),'Error in Petrosian Circular Radius (pixels)')
            self.morph_hdu.header['RPE']=(round(self.rp_ellip,8),'Petrosian Elliptical Semi-Major Axis (pixels)')
            self.morph_hdu.header['RPE_ERR']=(round(self.rp_ellip_err,8),'Error in Petrosian Elliptical Semi-Major Axis (pixels)')
            self.morph_hdu.header['ELONG']=(round(self.elongation,8),'Elongation Used in Elliptical Calcs')
            self.morph_hdu.header['ORIENT']=(round(self.pa_radians,8),'Orientation of ellipse in radians')
            self.morph_hdu.header['AXC']=(round(self.xcen_a2,8),'center x value minimizing Asym (pixels)')
            self.morph_hdu.header['AYC']=(round(self.ycen_a2,8),'center y value minimizing Asym (pixels)')
            self.morph_hdu.header['ASYM']=(round(self.asym2,8), 'Value of Asymmetry at final Asym center')

            #half light radii??



            #solve for moment center and segmap with a 2-step iteration
            #1.  Assume asymmetry center, compute segmap
            #2.  compute centroid in this initial segmap
            #3.  Compute segmap around this centroid instead
            #4.  Re-compute centroid?

            self.petro_segmap_init = self.petro_sma_segmap(self.xcen_a2,self.ycen_a2,self.rp_ellip)
            if self.petro_segmap_init is None:
                self.lotz_morph_status = 'Error: Poor RPA segmap (1) (probably negative avg flux)'
                print '        Exiting Morph calculation with status: '+self.lotz_morph_status
                return
            else:
                self.lotz_morph_status = 'Computed RPA segmap (1)'

            self.rpaseg_galaxy_image_init = np.where(self.petro_segmap_init==10.0, self.galaxy_image, np.zeros_like(self.galaxy_image))

            #find G-M20 center by minimizing 2nd moment
            #I'm pretty sure this is just the image centroid given the appropriate segmap???
            m00 = np.sum(self.rpaseg_galaxy_image_init)
            moment_array = (self.pixel_xpos)*self.rpaseg_galaxy_image_init
            self.mxc= np.sum(moment_array)/m00
            moment_array = (self.pixel_ypos)*self.rpaseg_galaxy_image_init
            self.myc= np.sum(moment_array)/m00
            #this assumes that 2nd moment is minimized when center is the centroid


            self.petro_segmap = self.petro_sma_segmap(self.mxc,self.myc,self.rp_ellip)
            self.rpaseg_galaxy_image = np.where(self.petro_segmap==10.0, self.galaxy_image, np.zeros_like(self.galaxy_image))
            if self.petro_segmap is None:
                self.lotz_morph_status = 'Error: Poor RPA segmap (2) (probably negative avg flux)'
                print '        Exiting Morph calculation with status: '+self.lotz_morph_status
                return
            else:
                self.lotz_morph_status = 'Computed RPA segmap (2)'

            m00 = np.sum(self.rpaseg_galaxy_image)
            moment_array = (self.pixel_xpos)*self.rpaseg_galaxy_image
            self.mxc= np.sum(moment_array)/m00
            moment_array = (self.pixel_ypos)*self.rpaseg_galaxy_image
            self.myc= np.sum(moment_array)/m00
            self.morph_hdu.header['MXC']=(round(self.mxc,8),'Centroid of final RPA image')
            self.morph_hdu.header['MYC']=(round(self.myc,8),'Centroid of final RPA image')


            #check S/N per galaxy pixel inside aperture defined above
            self.snpix = self.sn_per_pixel(self.galaxy_image,self.petro_segmap)
            self.morph_hdu.header['SNP']=(round(self.snpix,8),'Final average S/N per pixel')

            if self.snpix < 2.0:
                self.lotz_morph_status = 'Error: too faint!'
                print '        Exiting Morph calculation with status: '+self.lotz_morph_status
                print '                        Average S/N per pixel: {:7.3f}'.format(self.snpix)

                return
            else:
                self.lotz_morph_status = 'OK signal-to-noise (2)'


            #compute concentration--choose center?
            self.cc,self.cc_err,self.r20,self.r20_err,self.r80,self.r80_err,self.cc_status = self.concentration(self.xcen_a2,self.ycen_a2,1.5*self.rp_circ_2)
            if self.cc_status != 'Calculated C':
                self.lotz_morph_status = 'Warning: Bad Measurement of C'
                self.morph_hdu.header['CFLAG']=(1,'Bad C measurement')
                self.cflag = 1
                print '        Found bad concentration            : '+self.lotz_morph_status + '  '+self.cc_status, self.cc, self.cc_err, self.r20,self.r20_err,self.r80,self.r80_err
            else:
                self.lotz_morph_status = 'Computed C'
                self.morph_hdu.header['CFLAG']=(0,'OK C measurement')
                self.morph_hdu.header['CC']=(round(self.cc,8),'Concentration Parameter')
                self.morph_hdu.header['CC_ERR']=(round(self.cc_err,8),'Statistical error on Concentration Parameter')
                self.morph_hdu.header['CC_R20']=(round(self.r20,8),'R20 value (pixels)')
                self.morph_hdu.header['CC_R20E']=(round(self.r20_err,8),'R20 error (pixels)')
                self.morph_hdu.header['CC_R80']=(round(self.r80,8),'R80 value (pixels)')
                self.morph_hdu.header['CC_R80E']=(round(self.r80_err,8),'R80 error (pixels)')
                self.cflag = 0
                print '        Found concentration                :  {:8.4f}  {:8.4f}  {:8.4f}  {:8.4f}  {:8.4f}  {:8.4f}  {:25s}  '.format(self.cc, self.cc_err,self.r20,self.r20_err,self.r80,self.r80_err, self.cc_status)

            #use region from internal segmentation map algorithm
            analyze_image = self.rpaseg_galaxy_image

            #gini
            self.gini = self.compute_gini(self.galaxy_image,self.petro_segmap)
            if True:
                print '        Found Gini                         :  {:8.4f}   '.format(self.gini)
                self.lotz_morph_status = 'Computed G'
                self.morph_hdu.header['GINI']=(round(self.gini,8),'Gini (as defined by Lotz et al. 2004)')

            #m20
            self.m20 = self.compute_m20(analyze_image)
            if self.m20 is not None:
                print '        Found M20                          :  {:8.4f}   '.format(self.m20)
                self.lotz_morph_status = 'Computed M20'
                self.morph_hdu.header['M20']=(round(self.m20,8),'M20 (as defined by Lotz et al. 2004)')
            else:
                print '        Bad M20 (too faint or small?)      : ', np.sum(analyze_image)
                self.lotz_morph_status = 'Computed M20'                

            label,num=scipy.ndimage.measurements.label(self.petro_segmap)
            if num != 1:
                print '        Non-contiguous or non-existent segmap! {:4d}  '.format(num)
                self.lotz_morph_status = 'Error: bad segmap'



            #half-light radii
            self.rhalf_circ,self.rhalf_circ_err,self.rhalf_circ_status = self.fluxrad_circ(0.5,self.xcen_a2,self.ycen_a2,1.5*self.rp_circ_2)
            self.rhalf_ellip,self.rhalf_ellip_err,self.rhalf_ellip_status = self.fluxrad_ellip(0.5,self.xcen_a2,self.ycen_a2,1.5*self.rp_ellip)

            if self.rhalf_circ is not None and self.rhalf_ellip is not None:
                print '        Found R_half                       :  {:8.4f}   {:8.4f}  '.format(self.rhalf_circ,self.rhalf_circ_err)
                print '        Found R_half_e                     :  {:8.4f}   {:8.4f}  '.format(self.rhalf_ellip,self.rhalf_ellip_err)
                self.morph_hdu.header['R5C']=(round(self.rhalf_circ,8),'Circular half-light radius (pix)')
                self.morph_hdu.header['R5E']=(round(self.rhalf_ellip,8),'Elliptical half-light radius (pix)')
                self.morph_hdu.header['ER5C']=(round(self.rhalf_circ_err,8),'Error on circular half-light radius')
                self.morph_hdu.header['ER5E']=(round(self.rhalf_ellip_err,8),'Error on elliptical half-light radius')



            #MID
            #First, follow Freeman by assuming the galaxy lies at the very center of the image
            #Second, use Lotz petro segmap to compute everything a 2nd time, for comparison purposes
            #this alleviates some issues if segmaps are wildly different (can happen in mergers/clusters)
            self.midmap,self.sn_mid_center,self.midseg_area = self.mid_segmap(self.galaxy_image,int(float(self.npix)/2.0),int(float(self.npix)/2.0))
            if np.sum(self.midmap)==0.0:
                print '        Empty MID segmap!                             '
                self.lotz_morph_status = 'Error: Empty MID Segmap'
                return
            else:
                print '        Found MID segmap                              '

            #multiply by 1-valued MID segmap to get image for MID
            self.mid_image = self.galaxy_image*self.midmap
            #set negative values == 0
            self.mid_image = np.where(self.mid_image > 0.0, self.mid_image, np.zeros_like(self.mid_image))


            self.m_prime, self.m_stat_a1, self.m_stat_a2, self.m_stat_level = self.compute_m_statistic(self.mid_image)
            if self.m_prime is not None:
                print '        Found M statistic(1)               :  {:8.4f}     {:8.4f}    {:8.4f}    {:8.4f} '.format(self.m_prime,self.m_stat_a1,self.m_stat_a2,self.m_stat_level)
            else:
                print '        Bad M statistic (check mid segmap) :  '
                self.lotz_morph_status = 'Error: Bad M Statistic'
                return

            self.i_stat, self.i_stat_xpeak, self.i_stat_ypeak, self.i_stat_clump = self.compute_i_statistic(self.mid_image)
            #print self.i_stat, self.i_stat_xpeak, self.i_stat_ypeak, self.i_stat_clump.shape
            if self.i_stat is not None:
                print '        Found I statistic(1)               :  {:8.4f}     {:8.4f}    {:8.4f} '.format(self.i_stat, self.i_stat_xpeak, self.i_stat_ypeak)
            else:
                print '        Bad I statistic                    :  '
                self.lotz_morph_status = 'Error: Bad I Statistic'
                return

            self.d_stat, self.d_stat_area, self.d_stat_xcen, self.d_stat_ycen = self.compute_d_statistic(self.mid_image,self.i_stat_xpeak,self.i_stat_ypeak)
            if self.d_stat is not None:
                print '        Found D statistic(1)               :  {:8.4f}     {:8.4f}    {:8.4f}   {:8.4f}'.format(self.d_stat, self.d_stat_area, self.d_stat_xcen, self.d_stat_ycen)
            else:
                print '        Bad D statistic                    :  '
                self.lotz_morph_status = 'Error: Bad D Statistic'
                return

            self.mid1_snpix = self.sn_per_pixel(self.galaxy_image,self.midmap)


            self.lotz_morph_status = 'Computed MID (1)'
            self.morph_hdu.header['MIDSEG']=('Freeman','MID_ cards use Freeman segmap algo')
            self.morph_hdu.header['MID_AREA'] = (round(np.sum(self.midmap),1),'Area in pixels of MID segmap')
            self.morph_hdu.header['MID_SNP']=(round(self.mid1_snpix,8),'MID average S/N per pixel')
            self.morph_hdu.header['MID_MP']=(self.m_prime,'Mprime stat (Freeman et al. 2013)')
            self.morph_hdu.header['MID_A1']=(round(self.m_stat_a1,4),'Area 1 for Mprime')
            self.morph_hdu.header['MID_A2']=(round(self.m_stat_a2,4),'Area 2 for Mprime')
            self.morph_hdu.header['MID_LEV']=(round(self.m_stat_level,8),'Level for Mprime')
            self.morph_hdu.header['MID_I']=(self.i_stat,'I stat (Freeman et al. 2013)')
            self.morph_hdu.header['MID_IXP']=(round(self.i_stat_xpeak,2),'xpeak for I')
            self.morph_hdu.header['MID_IYP']=(round(self.i_stat_ypeak,2),'ypeak for I')
            self.morph_hdu.header['MID_D']=(self.d_stat,'D stat (Freeman et al. 2013)')
            self.morph_hdu.header['MID_DA']=(round(self.d_stat_area,4),'area for D stat')
            self.morph_hdu.header['MID_DXC']=(round(self.d_stat_xcen,4),'centroid for D stat')
            self.morph_hdu.header['MID_DYC']=(round(self.d_stat_ycen,4),'centroid for D stat')
            #we can also do fun M20, Gini here
            #use region from Freeman segmentation map algorithm

            #gini
            self.mid1_gini = self.compute_gini(self.galaxy_image,self.midmap*10.0)
            if True:
                print '        Found Gini(MID1)                   :  {:8.4f}   '.format(self.mid1_gini)
                self.lotz_morph_status = 'Computed G(2)'
                self.morph_hdu.header['MID_GINI']=(round(self.mid1_gini,8),'Gini in Freeman segmap')

            #m20
            self.mid1_m20 = self.compute_m20(self.galaxy_image*self.midmap)
            if self.mid1_m20 is not None:
                print '        Found M20(MID1)                    :  {:8.4f}   '.format(self.mid1_m20)
                self.lotz_morph_status = 'Computed M20(2)'
                self.morph_hdu.header['MID_M20']=(round(self.mid1_m20,8),'M20 in Freeman segmap')
            else:
                print '        Bad M20 (too faint or small?)      : ', np.sum(self.galaxy_image*self.midmap)
                self.lotz_morph_status = 'Computed M20(2)'    





            #try again with Lotz segmap for comparison
            #interesting note: trying Freeman segmap algo with Lotz center instead of galaxy center yields very similar segmap to Lotz
            self.mid2_image = 1.0*self.rpaseg_galaxy_image
            self.mid2_image = np.where(self.mid2_image > 0.0,self.mid2_image,np.zeros_like(self.mid2_image))


            self.mid2_m_prime, self.mid2_m_stat_a1, self.mid2_m_stat_a2, self.mid2_m_stat_level = self.compute_m_statistic(self.mid2_image)
            if self.mid2_m_prime is not None:
                print '        Found M statistic(2)               :  {:8.4f}     {:8.4f}    {:8.4f}    {:8.4f} '.format(self.mid2_m_prime,self.mid2_m_stat_a1,self.mid2_m_stat_a2,self.mid2_m_stat_level)
            else:
                print '        Bad M statistic(2)(check segmap)   :  '
                self.lotz_morph_status = 'Error: Bad M Statistic'
                return

            self.mid2_i_stat, self.mid2_i_stat_xpeak, self.mid2_i_stat_ypeak, self.mid2_i_clump = self.compute_i_statistic(self.mid2_image)
            if self.mid2_i_stat is not None:
                print '        Found I statistic(2)               :  {:8.4f}     {:8.4f}    {:8.4f} '.format(self.mid2_i_stat, self.mid2_i_stat_xpeak, self.mid2_i_stat_ypeak)
            else:
                print '        Bad I statistic(2)                 :  '
                self.lotz_morph_status = 'Error: Bad I Statistic'
                return

            self.mid2_d_stat, self.mid2_d_stat_area, self.mid2_d_stat_xcen, self.mid2_d_stat_ycen = self.compute_d_statistic(self.mid2_image,self.mid2_i_stat_xpeak,self.mid2_i_stat_ypeak)
            if self.mid2_d_stat is not None:
                print '        Found D statistic(2)               :  {:8.4f}     {:8.4f}    {:8.4f}   {:8.4f}'.format(self.mid2_d_stat, self.mid2_d_stat_area, self.mid2_d_stat_xcen, self.mid2_d_stat_ycen)
            else:
                print '        Bad D statistic(2)                 :  '
                self.lotz_morph_status = 'Error: Bad D Statistic'
                return

            self.lotz_morph_status = 'Computed MID (2)'
            self.morph_hdu.header['MID2SEG']=('Lotz','MID2_ cards use Lotz segmap algo')
            self.morph_hdu.header['MID2_MP']=(self.mid2_m_prime,'Mprime stat (Freeman et al. 2013)')
            self.morph_hdu.header['MID2_A1']=(round(self.mid2_m_stat_a1,4),'Area 1 for Mprime')
            self.morph_hdu.header['MID2_A2']=(round(self.mid2_m_stat_a2,4),'Area 2 for Mprime')
            self.morph_hdu.header['MID2_LEV']=(round(self.mid2_m_stat_level,8),'Level for Mprime')
            self.morph_hdu.header['MID2_I']=(self.mid2_i_stat,'I stat (Freeman et al. 2013)')
            self.morph_hdu.header['MID2_IXP']=(round(self.mid2_i_stat_xpeak,2),'xpeak for I')
            self.morph_hdu.header['MID2_IYP']=(round(self.mid2_i_stat_ypeak,2),'ypeak for I')
            self.morph_hdu.header['MID2_D']=(self.mid2_d_stat,'D stat (Freeman et al. 2013)')
            self.morph_hdu.header['MID2_DA']=(round(self.mid2_d_stat_area,4),'area for D stat')
            self.morph_hdu.header['MID2_DXC']=(round(self.mid2_d_stat_xcen,4),'centroid for D stat')
            self.morph_hdu.header['MID2_DYC']=(round(self.mid2_d_stat_ycen,4),'centroid for D stat')




        else:
            #raise error
            self.lotz_morph_status = 'Error: Uninitialized'

        print '        Exiting Morph calculation with status: '+self.lotz_morph_status
        return


    #following 5 functions adapted from Freeman et al. (2013) and Peth et al. (2016)
    def compute_d_statistic(self,img,xpeak,ypeak):
        nx = img.shape[0]
        ny = img.shape[1]
        xcen = 0.0
        ycen = 0.0

        #first, find centroids
        m00 = np.sum(img)
        moment_array = (self.pixel_xpos)*img
        xcen = np.sum(moment_array)/m00
        moment_array = (self.pixel_ypos)*img
        ycen= np.sum(moment_array)/m00
        area = np.sum(np.where(img > 0.0,np.ones_like(img),np.zeros_like(img)))

        d_stat = (((xpeak-xcen)**2 + (ypeak-ycen)**2)**0.5)/((area/math.pi)**0.5)

        return d_stat, area, xcen, ycen

    def compute_i_statistic(self,img,scale=1.0):
        if scale > 0.0:
            new_img = scipy.ndimage.filters.gaussian_filter(img,scale,mode='nearest')
        else:
            new_img = 1.0*img
            
        nx = new_img.shape[0]
        ny = new_img.shape[1]
        cimg = new_img*1.0
        clump,xpeak,ypeak = self.i_clump(new_img)

        w = np.where(xpeak != -9)[0]
        if w.shape[0] == 0:
            return None, None, None, clump
        elif w.shape[0]==1:
            return 0.0,xpeak[w][0]+0.5,ypeak[w][0]+0.5,clump
        else:
            #w.shape[0] > 1
            int_clump = np.zeros_like(np.float32(w))
            ucl = np.unique(clump)
            for cv in ucl:
                if cv<=0:
                    continue
                int_clump[cv-1]=np.sum(np.where(clump==cv,img,np.zeros_like(img)))

            #I think the above implementation is way faster?
            #for jj in np.arange(nx):
            #    for kk in np.arange(ny):
            #        if clump[jj,kk] > 0:
            #            int_clump[clump[jj,kk]-1]=int_clump[clump[jj,kk]-1] + img[jj,kk]
            mxi = np.argmax(int_clump)
            mx = int_clump[mxi]
            xpeak = xpeak[mxi]
            ypeak = ypeak[mxi]
            s = np.argsort(int_clump)
            int_ratio = int_clump[s][-2]/int_clump[s][-1]
            return int_ratio,xpeak+0.5,ypeak+0.5,clump  #registering center on same pixel grid for D

    def i_clump(self,img):
        nx = img.shape[0]
        ny = img.shape[1]
        clump = -1 + np.zeros_like(np.int32(img))
        xpeak = -9 + np.zeros_like(np.linspace(0,1,100))
        ypeak = -9 + np.zeros_like(np.linspace(0,1,100))
        for jj in np.arange(nx):
            for kk in np.arange(ny):
                if img[jj,kk]==0.0:
                    continue
                jjcl=jj*1
                kkcl=kk*1
                istop=0
                while (istop==0):
                    jjmax=jjcl*1
                    kkmax=kkcl*1
                    imgmax=img[jjcl,kkcl]
                    for mm in [-1,0,1]:
                        if (jjcl+mm >= 0) and (jjcl+mm < nx):
                            for nn in [-1,0,1]:
                                if (kkcl+nn >= 0) and (kkcl+nn < ny):
                                    if (img[jjcl+mm,kkcl+nn] > imgmax):
                                        imgmax = img[jjcl+mm,kkcl+nn]
                                        jjmax=jjcl+mm
                                        kkmax=kkcl+nn
                    #end of mm, nn loops
                    if jjmax==jjcl and kkmax==kkcl:
                        ifound=0
                        mm=0
                        while (ifound==0) and (xpeak[mm] != -9) and (mm < 99):
                            if (xpeak[mm]==jjmax) and (ypeak[mm]==kkmax):
                                ifound=1
                            else:
                                mm = mm+1
                        #endwhile
                        if (ifound==0):
                            xpeak[mm]=jjmax
                            ypeak[mm]=kkmax
                        clump[jj,kk]=mm
                        istop=1
                    else:
                        jjcl = jjmax
                        kkcl = kkmax
                #endwhile
            #endfor
        #endfor
        clump = clump+1

        return clump, xpeak, ypeak   

    def compute_m_statistic(self,img,levels=None):
        if levels is None:
            levels = np.linspace(0.5,0.98,num=25)

        nlevels = levels.shape[0]
        norm_img = img/np.max(img)
        area_ratio = np.zeros_like(levels)
        a1 = np.zeros_like(levels)
        a2 = np.zeros_like(levels)
        max_level = 0
        w = np.where(norm_img != 0.0)
        if w[0].shape[0]==0:
            return None,None,None,None

        npix = w[0].shape[0]

        snorm_img = norm_img[w]
        si = np.argsort(snorm_img)
        snorm_img = snorm_img[si]  #sorted non-zero pixels
        ai = np.argsort(norm_img)

        for i,lev in enumerate(levels):
            thr = round(npix*lev)-1
            w = np.where(norm_img >= snorm_img[thr])
            if w[0].shape[0] > 0:
                thr_arr = np.asarray([snorm_img[thr],1.0])
                r,num,clump = region_grow.region_grow(norm_img,ai,THRESHOLD=thr_arr)
                if r.shape[0] > 1:
                    u,counts = np.unique(clump,return_counts=True)
                    nzi = np.where(u != 0.0)[0]

                    if nzi.shape[0] > 1:
                        new_u = u[nzi]
                        new_counts = counts[nzi]
                        sci = np.argsort(new_counts)

                        a1[i] = float(new_counts[sci[-1]])#area of largest clump
                        a2[i] = float(new_counts[sci[-2]])#area of 2nd largest
                        area_ratio[i] = float(a2[i])/float(a1[i])
                    
        if np.max(area_ratio) > 0.0:
            imax = np.argmax(area_ratio)
            max_level = levels[imax]
            m_prime = area_ratio[imax]
            a1_val = a1[imax]
            a2_val = a2[imax]
            return m_prime, a1_val, a2_val, max_level
        else:
            return 0.0,0.0,0.0,0.0

    def mid_segmap(self, img, xcen, ycen, e = 0.2, t = 100.0):
        flat_img = img.flatten()
        si = np.argsort(flat_img)
        sort_img = flat_img[si]
        npix = sort_img.shape[0]

        minval = np.min(img)
        maxval = np.max(img)

        #level = np.logspace(np.log10(0.99),-5.0,num=198)
        #expanded number of quantiles... was going down too low with only 200
        level = np.linspace(0.999,0.0,num=2000)
        nlevel = level.shape[0]
        mid_segmap = np.zeros_like(img)
        mu = 0.0
        dmu = 0.0
        dnw=0

        sn_mid_center = img[xcen,ycen]/self.skysig

        for i,lev in enumerate(level):
            thr = np.asarray([sort_img[int(lev*npix)], np.max(img)])
            r,num,clump = region_grow.region_grow(img,si,THRESHOLD=thr)
            if clump[xcen,ycen]==0.0:
                continue
            w = np.where(clump==clump[xcen,ycen])
            if mu > 0.0:
                dnw = w[0].shape[0]-nw
                if dnw < 16:
                    continue
                dmu = (np.sum(img[w])-mu*nw)/float(dnw)
                #nw = w[0].shape[0]
                #print i,lev,r.shape, xcen, ycen, clump[xcen,ycen], thr, self.skysig,nw, dnw
                if (dnw > 1.1*npix/2000.0) and (i > t-1):
                    nthr = np.asarray([sort_img[int(level[i-1]*npix)], np.max(img)])
                    r,num,clump = region_grow.region_grow(img,si,THRESHOLD=nthr)
                    w = np.where(clump==clump[xcen,ycen])
                    dnw = w[0].shape[0]-nw
                    if dnw < 16:
                        continue
                    dmu = (np.sum(img[w])-mu*nw)/float(dnw)
                    #nw = w[0].shape[0]
                if (dmu/(mu+dmu) < e):
                    mid_segmap = np.where(clump==clump[xcen,ycen],np.ones_like(clump),np.zeros_like(clump))
                    #regularize map
                    #mid_segmap = scipy.ndimage.filters.uniform_filter(mid_segmap,size=5)
                    mid_segmap = scipy.ndimage.filters.gaussian_filter(mid_segmap,self.psf_fwhm_pixels/2.355)
                    mid_segmap = np.where(mid_segmap > 1.0e-1,np.ones_like(mid_segmap),np.zeros_like(mid_segmap))
                    #print '{:8d}  {:5.4f}  {:10.4f}  {:10.4f}  {:8d}'.format(i, lev, mu, dmu, dnw)
                    return mid_segmap,sn_mid_center,np.sum(mid_segmap)
                    
            mu = np.mean(img[w])
            nw = w[0].shape[0]
            #print '{:8d}  {:5.4f}  {:10.4f}  {:10.4f}  {:8d}'.format(i, lev, mu, dmu, dnw)

        return mid_segmap,sn_mid_center,np.sum(mid_segmap)



    #estimate mean S/N ratio of galaxy pixels, assuming long integration times (sky dominated)
    def sn_per_pixel(self,image,segmap):
        im = image.flatten()
        seg = segmap.flatten()
        ap = np.where(seg > 0.0)[0]
        n = np.sum(np.ones_like(ap))
        print n, self.skysig
        s2n = np.sum( im[ap]/((self.skysig**2)**0.5))/n

        return s2n

    #calculate M_20 as in Lotz et al. 2004
    def compute_m20(self,analyze_image):
        m20 = 0.0


        mu,eta = central_moments(analyze_image,[2,0],[0,2],xc=self.mxc,yc=self.myc)
        mtot = mu[0] + mu[1]
        
        x_array = (self.pixel_xpos).flatten()-self.mxc
        y_array = (self.pixel_ypos).flatten()-self.myc
        r2_array = x_array**2 + y_array**2
        im_array = analyze_image.flatten()
        si = np.flipud(np.argsort(im_array))

        fsum = 0.0
        mom20 = 0.0
        totsum = np.sum(im_array)

        for i in si:
            fsum = fsum + im_array[i]
            mom20 = mom20 + im_array[i]*r2_array[i]
            if fsum/totsum > 0.20:
                break

        if mom20 > 0.0 and mtot > 0.0:
            m20param = np.log10(mom20/mtot)
            return m20param
        else:
            return None

    #calculate Gini as in Lotz et al. 2004
    def compute_gini(self,analyze_image,segmap):
        analyze_image = (analyze_image).flatten()
        map_pixels = (segmap).flatten()

        pixelvals = np.abs( (analyze_image)[np.where(map_pixels == 10.0)[0]] )
        sorted_pixelvals = np.sort(pixelvals)
        total_absflux = np.sum(sorted_pixelvals)
        mean_absflux = np.mean(sorted_pixelvals)
        gini = 0.0
        n = float(sorted_pixelvals.shape[0])

        for i,x in enumerate(sorted_pixelvals):
            gini = gini + (2.0*float(i)-n-1.0)*x

        gini = gini/total_absflux/(n-1.0)

        return gini

    #calculate circular concentration following Conselice 2003
    #Note: experiments reveal a difference between this algorithm
    #      and results using Lotz et al. (2004) IDL code, 
    #      This code gives ~0.2 lower median C values with a 
    #          random difference of ~0.2 compared with IDL version 
    #      Reason: IDL implementation unconverged in center of curve-of-growth
    #              Requires smaller/more accurate sub-pixel steps
    #              Python version uses effectively infinite pixel resolution
    #              and is shown to be stable wrt number of points in COG
    def concentration(self,xcenter,ycenter,extent):
        xi = np.float32(np.arange(self.npix+1))
        xpos,ypos = np.meshgrid(xi,xi)
        xmin,xmax = np.min(xpos-xcenter),np.max(xpos-xcenter)
        ymin,ymax = np.min(ypos-ycenter),np.max(ypos-ycenter)
        analyze_image = self.galaxy_image

        frac_overlap_extent = photutils.geometry.circular_overlap_grid(xmin,xmax,ymin,ymax,self.npix,self.npix,extent,1,3)
        total_area = np.sum(frac_overlap_extent)
        total_flux = np.sum(analyze_image*frac_overlap_extent)

        if total_flux <= 0.0:
            return None,None,None,None,None,None,'Error: Probably too faint/weird to measure C'

        err_on_total = ((total_area)*self.skysig**2)**0.5

        numpts=200
        minr = 0.2
        maxr = extent
        radius_grid = np.logspace(np.log10(minr),np.log10(maxr),num=numpts)
        #radius_grid = np.linspace(minr,maxr,num=numpts)

        r20 = 0.0
        r20_err = -1.0
        r80 = 0.0
        r80_err = -1.0
        this_flux = np.zeros_like(radius_grid)
        ffrac = np.zeros_like(radius_grid)
        ffrac_err = np.zeros_like(radius_grid)



        for i,r in enumerate(radius_grid):
            frac_overlap_r = photutils.geometry.circular_overlap_grid(xmin,xmax,ymin,ymax,self.npix,self.npix,r,1,3)
            area = np.sum(frac_overlap_r)
            #expected_negative_flux = -1.0*self.skysig*(area/2.0)*(math.pi/4.0)

            this_flux[i] = np.sum(analyze_image*frac_overlap_r)
            ffrac[i] = this_flux[i]/total_flux
            
            err_on_flux = ((area)*self.skysig**2)**0.5  #error on average flux

            if (this_flux[i] > 0.0):
                ffrac_err[i] = ( (err_on_flux/this_flux[i])**2 + (err_on_total/total_flux)**2 )**0.5
            else:
                #avg_flux_in_r is negative
                ffrac_err[i] = -1.0


            #print ffrac[i], r, ffrac_err[i], total_flux

        #evaluate curve of growth for 0.2, 0.8
        r20,r20_err,r20_status = self.evaluate_cog(0.2,radius_grid,this_flux,ffrac,ffrac_err,total_flux)
        r80,r80_err,r80_status = self.evaluate_cog(0.8,radius_grid,this_flux,ffrac,ffrac_err,total_flux)

        #print r20, r20_err, r20_status
        #print r80, r80_err, r80_status

        if (r20_status == 'Positive R_val') and (r80_status == 'Positive R_val'):
            #compute concentration, return all
            assert r20 is not None
            assert r80 is not None
            assert r20_err > 0.0
            assert r80_err > 0.0

            if r20 < r80:
                #this is a correct result
                c = 5.0*np.log10(r80/r20)
                dcdr80 = 5.0*(1.0/r80)*(1.0/np.log(10.0))
                dcdr20 = -5.0*(1.0/r20)*(1.0/np.log(10.0))
                c_err = ( (dcdr80**2)*(r80_err**2) + (dcdr20**2)*(r20_err**2) )**0.5
                return c, c_err, r20, r20_err, r80, r80_err, 'Calculated C'

            else:
                #this is weird
                return None,None,None,None,None,None,'Error:Weird C calculation'
        else:
            #error, return Nones and status
            return None,None,None,None,None,None,'Error: r20 or r80 poorly defined'


    #helper function for concentration and other curve of growth estimates
    def evaluate_cog(self,value, r_array, flux_r, frac_r, fracerr_r, totalflux):
        this_r_pixels = None
        r_pixels_err = None

        if np.min(frac_r) > value:
            status = 'Galaxy nucleus dominates'
        elif np.max(frac_r) < value:
            #I think this will most often happen when all are zero
            status = 'Galaxy too small or faint'
        else:
            #max is > 0.2 and min is < 0.2 --> it crosses at least once
            #find first crossing and then interpolate
            #demand that first crossing occurs after r=0.2 pix otherwise we're probably just seeing noise
            eta_ind = np.where(np.logical_and(frac_r >= value,r_array > 0.2))[0]
            if eta_ind.shape[0] > 0:
                for ei in eta_ind:
                    if ei==0:
                        continue
                    elif frac_r[ei-1] < value:
                        #interpolate
                        this_r_pixels = np.interp(value,frac_r[ei-1:ei+1],r_array[ei-1:ei+1])
                        delta_eta = np.abs(frac_r[ei-1] - frac_r[ei])
                        delta_r = np.abs(r_array[ei-1] - r_array[ei])
                        fsigma_eta = np.max(fracerr_r[ei-1:ei+1])
                        sigma_eta = value*fsigma_eta
                        r_pixels_err = (((delta_r/delta_eta)**2)*(sigma_eta**2))**0.5  #conservative estimate of error on r

                        #print eta_ind
                        #print value, ei, frac_r[ei-1:ei+1], r_array[ei-1:ei+1]
                        #print this_r_pixels, delta_eta, delta_r, fsigma_eta, sigma_eta, r_pixels_err
                        break
                    else:
                        continue
                if this_r_pixels<=0.0:
                    status = 'Weird light profile or too faint'
                else:
                    status = 'Positive R_val'
            else:
                #actually it never crosses: error
                status = 'Error: problem with R_val curve of growth'

        return this_r_pixels, r_pixels_err, status



    def fluxrad_circ(self,fluxfrac,xcenter,ycenter,extent):
        xi = np.float32(np.arange(self.npix+1))
        xpos,ypos = np.meshgrid(xi,xi)
        xmin,xmax = np.min(xpos-xcenter),np.max(xpos-xcenter)
        ymin,ymax = np.min(ypos-ycenter),np.max(ypos-ycenter)
        analyze_image = self.galaxy_image

        frac_overlap_extent = photutils.geometry.circular_overlap_grid(xmin,xmax,ymin,ymax,self.npix,self.npix,extent,1,3)
        total_area = np.sum(frac_overlap_extent)
        total_flux = np.sum(analyze_image*frac_overlap_extent)

        if total_flux <= 0.0:
            return None,None,None,None,None,None,'Error: Probably too faint/weird to measure C'

        err_on_total = ((total_area)*self.skysig**2)**0.5

        numpts=200
        minr = 0.2
        maxr = extent
        radius_grid = np.logspace(np.log10(minr),np.log10(maxr),num=numpts)
        #radius_grid = np.linspace(minr,maxr,num=numpts)

        r20 = 0.0
        r20_err = -1.0
        r80 = 0.0
        r80_err = -1.0
        this_flux = np.zeros_like(radius_grid)
        ffrac = np.zeros_like(radius_grid)
        ffrac_err = np.zeros_like(radius_grid)

        for i,r in enumerate(radius_grid):
            frac_overlap_r = photutils.geometry.circular_overlap_grid(xmin,xmax,ymin,ymax,self.npix,self.npix,r,1,3)
            area = np.sum(frac_overlap_r)

            this_flux[i] = np.sum(analyze_image*frac_overlap_r)
            ffrac[i] = this_flux[i]/total_flux
            
            err_on_flux = ((area)*self.skysig**2)**0.5  #error on average flux

            if (this_flux[i] > 0.0):
                ffrac_err[i] = ( (err_on_flux/this_flux[i])**2 + (err_on_total/total_flux)**2 )**0.5
            else:
                #avg_flux_in_r is negative
                ffrac_err[i] = -1.0

        rf,rf_err,rf_status = self.evaluate_cog(fluxfrac,radius_grid,this_flux,ffrac,ffrac_err,total_flux)

        return rf,rf_err,rf_status


    def fluxrad_ellip(self,fluxfrac,xcenter,ycenter,extent):
        xi = np.float32(np.arange(self.npix+1))
        xpos,ypos = np.meshgrid(xi,xi)
        xmin,xmax = np.min(xpos-xcenter),np.max(xpos-xcenter)
        ymin,ymax = np.min(ypos-ycenter),np.max(ypos-ycenter)
        analyze_image = self.galaxy_image

        frac_overlap_extent = photutils.geometry.elliptical_overlap_grid(xmin,xmax,ymin,ymax,self.npix,self.npix,extent,extent/self.elongation,self.pa_radians,1,3)
        total_area = np.sum(frac_overlap_extent)
        total_flux = np.sum(analyze_image*frac_overlap_extent)

        if total_flux <= 0.0:
            return None,None,None,None,None,None,'Error: Probably too faint/weird to measure C'

        err_on_total = ((total_area)*self.skysig**2)**0.5

        numpts=200
        minr = 0.2
        maxr = extent
        radius_grid = np.logspace(np.log10(minr),np.log10(maxr),num=numpts)
        #radius_grid = np.linspace(minr,maxr,num=numpts)

        r20 = 0.0
        r20_err = -1.0
        r80 = 0.0
        r80_err = -1.0
        this_flux = np.zeros_like(radius_grid)
        ffrac = np.zeros_like(radius_grid)
        ffrac_err = np.zeros_like(radius_grid)

        for i,r in enumerate(radius_grid):
            frac_overlap_r = photutils.geometry.elliptical_overlap_grid(xmin,xmax,ymin,ymax,self.npix,self.npix,r,r/self.elongation,self.pa_radians,1,3)
            area = np.sum(frac_overlap_r)

            this_flux[i] = np.sum(analyze_image*frac_overlap_r)
            ffrac[i] = this_flux[i]/total_flux
            
            err_on_flux = ((area)*self.skysig**2)**0.5  #error on average flux

            if (this_flux[i] > 0.0):
                ffrac_err[i] = ( (err_on_flux/this_flux[i])**2 + (err_on_total/total_flux)**2 )**0.5
            else:
                #avg_flux_in_r is negative
                ffrac_err[i] = -1.0

        rf,rf_err,rf_status = self.evaluate_cog(fluxfrac,radius_grid,this_flux,ffrac,ffrac_err,total_flux)

        return rf,rf_err,rf_status



    #code's internal segmentation map algorithm, following Lotz et al. (2004)
    #this is probably the most important uncertainty in the translation from the IDL code
    def petro_sma_segmap(self,xcenter,ycenter,r_ellip):
        #first, convolve by Gaussian with FWHM~ 1/5 (1/10?) petrosian radius?
        fwhm_pixels = r_ellip/10.0
        galaxy_psf_pixels = self.psf_fwhm_arcsec/self.pixelscale_arcsec

        s = 10
        area = float(s**2)

        #minimum smoothing length ~3x image PSF
        if fwhm_pixels < 3.0*galaxy_psf_pixels:
            fwhm_pixels = 3.0*galaxy_psf_pixels

        sigma_pixels = fwhm_pixels/2.355
        #use masked self.galaxy_image version
        smoothed_image = scipy.ndimage.filters.gaussian_filter(self.galaxy_image,sigma_pixels,mode='nearest')
        self.rpa_sigma_pixels = sigma_pixels

        #compute surface brightness at petrosian radius
        xi = np.float32(np.arange(self.npix+1))
        xpos,ypos = np.meshgrid(xi,xi)
        xmin,xmax = np.min(xpos-xcenter),np.max(xpos-xcenter)
        ymin,ymax = np.min(ypos-ycenter),np.max(ypos-ycenter)
        frac_overlap_rminus = photutils.geometry.elliptical_overlap_grid(xmin,xmax,ymin,ymax,self.npix,self.npix,r_ellip-1.0,(r_ellip-1.0)/self.elongation,self.pa_radians,1,3)
        frac_overlap_rplus = photutils.geometry.elliptical_overlap_grid(xmin,xmax,ymin,ymax,self.npix,self.npix,r_ellip+1.0,(r_ellip+1.0)/self.elongation,self.pa_radians,1,3)
        frac_overlap_annulus = frac_overlap_rplus - frac_overlap_rminus
        area_in_annulus = np.sum(frac_overlap_annulus)
        avg_flux_in_annulus = 0.0
        err_in_annulus = 0.0
        if area_in_annulus > 0.0:
            avg_flux_in_annulus = np.sum(smoothed_image*frac_overlap_annulus)/area_in_annulus
            err_in_annulus = ((1.0/area_in_annulus)*self.skysig**2)**0.5

        #do we also want a S/N test here?

        if avg_flux_in_annulus > 0.0:
            #set pixels with flux >= mu equal to 10, < mu equal to 0.0
            #initial calculation
            self.galaxy_smoothed_segmap = np.where(smoothed_image >= avg_flux_in_annulus,10.0*np.ones_like(smoothed_image),np.zeros_like(smoothed_image))

            #median filter to remove outlying pixels, useful mainly if rejecting CRs or bad pixels
            if self.filter_segmap:
                self.medfilt_segmap = scipy.ndimage.filters.uniform_filter(self.galaxy_smoothed_segmap,size=10,mode='constant',cval=0.0)-self.galaxy_smoothed_segmap/(100.0)
                self.stdfilt = scipy.ndimage.filters.generic_filter(self.galaxy_smoothed_segmap,std_value,size=10,mode='constant',cval=0.0)
                self.stdfilt = np.where(self.stdfilt > 0.0,self.stdfilt,np.zeros_like(self.stdfilt))

                self.filtered_segmap = np.where( np.abs(self.galaxy_smoothed_segmap - self.medfilt_segmap) > 3.0*self.stdfilt, self.medfilt_segmap, self.galaxy_smoothed_segmap)
            else:
                self.filtered_segmap = 1.0*self.galaxy_smoothed_segmap
                #issues with non-contiguous segmap get flagged later, or will have low S/N

            #finally, anything that survives with a nonzero value is part of the galaxy
            self.filtered_segmap = np.where(self.filtered_segmap > 0.001,10.0*np.ones_like(smoothed_image),np.zeros_like(smoothed_image))

            return self.filtered_segmap#galaxy_segmap #

        else:
            #uhh, problem
            return None



    def compute_asym(self,xcenter=None,ycenter=None,extent=None,a_bkg=None):

        assert xcenter is not None
        assert ycenter is not None
        assert extent is not None
        assert a_bkg is not None

        x0 = np.asarray([xcenter,ycenter])

        OptimizeResult = scipy.optimize.minimize(self._asymmetry_wrapper, x0, args=(extent,a_bkg), method='Powell',options={'disp': False, 'return_all': False, 'maxiter': 400, 'maxfev': None, 'xtol': 0.1, 'ftol': 0.01})
        if OptimizeResult.success:
            asym = OptimizeResult.fun
            xcen_a = OptimizeResult.x[0]
            ycen_a = OptimizeResult.x[1]
        else:
            asym=99.0
            xcen_a = xcenter
            ycen_a = ycenter

        message = OptimizeResult.message

        return float(asym), xcen_a, ycen_a, message

    def _asymmetry_wrapper(self,coords,radius=1.0,a_bkg=99.0):

        asym, ga, ba = self.galaxy_asymmetry(coords,radius=radius,a_bkg=a_bkg)

        return asym

    def galaxy_asymmetry(self,coords,radius=1.0,a_bkg=99.0):
        xc = coords[0]
        yc = coords[1]
        #don't let the asymmetry center wander off toward the edge of the image
        if xc<=radius or yc<=radius or xc>=self.npix-radius or yc>=self.npix-radius:
            asym=99.0
            a_gal = 99.0
            noise_offset = 99.0
            #print asym, xc, yc
        else:
            #following Lotz code from December 2013
            #must confirm

            analyze_image = self.galaxy_image

            rot_gal_im = skimage.transform.rotate(analyze_image,180.0,center=(xc,yc),mode='constant',cval=0.0,preserve_range=True)

            gal_dif = self.galaxy_image - rot_gal_im

            xmin = 0.0-xc
            xmax = self.npix+1 - xc
            ymin = 0.0-yc
            ymax = self.npix+1 - yc

            frac_overlap_r = photutils.geometry.circular_overlap_grid(xmin,xmax,ymin,ymax,self.npix,self.npix,radius,1,3)
            norm = np.sum(frac_overlap_r)  #area of galaxy


            gal_im_ap = analyze_image*frac_overlap_r
            gal_dif_ap = gal_dif*frac_overlap_r
            
            total_gal_ap = np.sum(np.abs(gal_im_ap))

            a_gal = np.sum(np.abs(gal_dif_ap))/total_gal_ap

            #effectively divides by the average intensity per pixel
            #a_bkg is already an area-normalized quantity (normed to background box area)
            noise_offset = norm*a_bkg/total_gal_ap

            asym = a_gal - noise_offset
            #print asym, a_gal, noise_offset, norm, total_gal_ap, a_bkg, xc, yc

        return asym, a_gal, noise_offset

    def rpetro_circ(self,xcenter=None,ycenter=None):
        numpts=100#self.npix
        #radius_grid = np.linspace(0.01,self.npix,num=numpts)
        minr = 1.5
        maxr = float(self.npix)/2.0
        radius_grid = np.logspace(np.log10(minr),np.log10(maxr),num=numpts)

        xi = np.float32(np.arange(self.npix+1))
        xpos,ypos = np.meshgrid(xi,xi)
        xmin,xmax = np.min(xpos-xcenter),np.max(xpos-xcenter)
        ymin,ymax = np.min(ypos-ycenter),np.max(ypos-ycenter)
        petro_ratio = np.zeros_like(radius_grid)
        petro_r_pixels = 0.0
        petro_r_pixels_err = -1.0
        petro_ratio_ferr = np.zeros_like(radius_grid)

        analyze_image = self.galaxy_image

        for i,r in enumerate(radius_grid):
            frac_overlap_r = photutils.geometry.circular_overlap_grid(xmin,xmax,ymin,ymax,self.npix,self.npix,r,1,3)
            frac_overlap_08r = photutils.geometry.circular_overlap_grid(xmin,xmax,ymin,ymax,self.npix,self.npix,r-1.0,1,3)
            frac_overlap_125r = photutils.geometry.circular_overlap_grid(xmin,xmax,ymin,ymax,self.npix,self.npix,r+1.0,1,3)
            
            area_in_r = np.sum(frac_overlap_r)

            avg_flux_in_r = np.sum(analyze_image*frac_overlap_r)/area_in_r
            err_in_r = ((1.0/area_in_r)*self.skysig**2)**0.5  #error on average flux

            frac_overlap_annulus = frac_overlap_125r - frac_overlap_08r
            area_in_annulus = np.sum(frac_overlap_annulus)
            avg_flux_in_annulus = 0.0
            err_in_annulus = 0.0
            if area_in_annulus > 0.0:
                avg_flux_in_annulus = np.sum(analyze_image*frac_overlap_annulus)/area_in_annulus
                err_in_annulus = ((1.0/area_in_annulus)*self.skysig**2)**0.5

            if (avg_flux_in_r > 0.0) and (avg_flux_in_annulus > 0.0):
                petro_ratio[i] = avg_flux_in_annulus/avg_flux_in_r
                petro_ratio_ferr[i] = ( (err_in_r/avg_flux_in_r)**2 + (err_in_annulus/avg_flux_in_annulus)**2 )**0.5
            elif (avg_flux_in_r > 0.0) and (avg_flux_in_annulus <= 0.0):
                petro_ratio[i] = 0.0
                petro_ratio_ferr[i] = -1.0
            else:
                #avg_flux_in_r is negative
                petro_ratio[i] = 0.0
                petro_ratio_ferr[i] = -1.0


            #if zeros for awhile, break
            if i > 25 and np.sum(petro_ratio[i-10:i+1])==0.0:
                break

        #evaluate RMS of r and annuli fluxes to estimate convergence? measure sigma_petro_ratio?
        

        #evaluate petrosian ratio curve
        if np.min(petro_ratio) > self.rpetro_eta:
            status = 'Galaxy too large'
        elif np.max(petro_ratio) < self.rpetro_eta:
            #I think this will most often happen when all are zero
            status = 'Galaxy too small or faint'
        else:
            #max is > 0.2 and min is < 0.2 --> it crosses at least once
            #find first crossing and then interpolate
            #demand that first crossing occurs after r=1.5 pix otherwise we're probably just seeing noise
            eta_ind = np.where(np.logical_and(petro_ratio <= self.rpetro_eta,radius_grid > 1.5))[0]
            if eta_ind.shape[0] > 0:
                for ei in eta_ind:
                    if ei==0:
                        continue
                    elif petro_ratio[ei-1] > self.rpetro_eta:
                        #interpolate
                        petro_r_pixels = np.interp(self.rpetro_eta,np.flipud(petro_ratio[ei-1:ei+1]),np.flipud(radius_grid[ei-1:ei+1]))
                        delta_eta = np.abs(petro_ratio[ei-1] - petro_ratio[ei])
                        delta_r = np.abs(radius_grid[ei-1] - radius_grid[ei])
                        fsigma_eta = np.max(petro_ratio_ferr[ei-1:ei+1])
                        sigma_eta = self.rpetro_eta*fsigma_eta
                        petro_r_pixels_err = (((delta_r/delta_eta)**2)*(sigma_eta**2))**0.5  #conservative estimate of error on rpetro estimate

                        #print petro_r_pixels, delta_eta, delta_r, fsigma_eta, sigma_eta, petro_r_pixels_err, petro_ratio[ei]
                        break
                    else:
                        continue
                if petro_r_pixels<=0.0:
                    status = 'Weird light profile or too faint'
                else:
                    status = 'Positive R_pet'
            else:
                #actually it never crosses: error
                status = 'Error: problem with R_pet curve of growth'

        return petro_r_pixels, status, petro_r_pixels_err






    def rpetro_ellip(self,xcenter=None,ycenter=None):
        numpts=100#self.npix
        #radius_grid = np.linspace(0.01,self.npix,num=numpts)
        minr = 1.5
        maxr = float(self.npix)/2.0
        radius_grid = np.logspace(np.log10(minr),np.log10(maxr),num=numpts)

        xi = np.float32(np.arange(self.npix+1))
        xpos,ypos = np.meshgrid(xi,xi)
        xmin,xmax = np.min(xpos-xcenter),np.max(xpos-xcenter)
        ymin,ymax = np.min(ypos-ycenter),np.max(ypos-ycenter)
        petro_ratio = np.zeros_like(radius_grid)
        petro_r_pixels = 0.0
        petro_r_pixels_err = -1.0
        petro_ratio_ferr = np.zeros_like(radius_grid)

        analyze_image = self.galaxy_image

        for i,r in enumerate(radius_grid):
            #r = semimajor axis
            ry = r/self.elongation #semiminor axis
            frac_overlap_r = photutils.geometry.elliptical_overlap_grid(xmin,xmax,ymin,ymax,self.npix,self.npix,r,ry,self.pa_radians,1,3)
            frac_overlap_08r = photutils.geometry.elliptical_overlap_grid(xmin,xmax,ymin,ymax,self.npix,self.npix,r-1.0,(r-1.0)/self.elongation,self.pa_radians,1,3)
            frac_overlap_125r = photutils.geometry.elliptical_overlap_grid(xmin,xmax,ymin,ymax,self.npix,self.npix,r+1.0,(r+1.0)/self.elongation,self.pa_radians,1,3)
            
            area_in_r = np.sum(frac_overlap_r)

            avg_flux_in_r = np.sum(analyze_image*frac_overlap_r)/area_in_r
            err_in_r = ((1.0/area_in_r)*self.skysig**2)**0.5  #error on average flux

            frac_overlap_annulus = frac_overlap_125r - frac_overlap_08r
            area_in_annulus = np.sum(frac_overlap_annulus)
            avg_flux_in_annulus = 0.0
            err_in_annulus = 0.0
            if area_in_annulus > 0.0:
                avg_flux_in_annulus = np.sum(analyze_image*frac_overlap_annulus)/area_in_annulus
                err_in_annulus = ((1.0/area_in_annulus)*self.skysig**2)**0.5

            if (avg_flux_in_r > 0.0) and (avg_flux_in_annulus > 0.0):
                petro_ratio[i] = avg_flux_in_annulus/avg_flux_in_r
                petro_ratio_ferr[i] = ( (err_in_r/avg_flux_in_r)**2 + (err_in_annulus/avg_flux_in_annulus)**2 )**0.5
            elif (avg_flux_in_r > 0.0) and (avg_flux_in_annulus <= 0.0):
                petro_ratio[i] = 0.0
                petro_ratio_ferr[i] = -1.0
            else:
                #avg_flux_in_r is negative
                petro_ratio[i] = 0.0
                petro_ratio_ferr[i] = -1.0


            #if zeros for awhile, break
            if i > 25 and np.sum(petro_ratio[i-10:i+1])==0.0:
                break

        #evaluate RMS of r and annuli fluxes to estimate convergence? measure sigma_petro_ratio?
        

        #evaluate petrosian ratio curve
        if np.min(petro_ratio) > self.rpetro_eta:
            status = 'Galaxy too large or unexpectedly elongated'
        elif np.max(petro_ratio) < self.rpetro_eta:
            #I think this will most often happen when all are zero
            status = 'Galaxy too small or faint'
        else:
            #max is > 0.2 and min is < 0.2 --> it crosses at least once
            #find first crossing and then interpolate
            #demand that first crossing occurs after r=1.5 pix otherwise we're probably just seeing noise
            eta_ind = np.where(np.logical_and(petro_ratio <= self.rpetro_eta,radius_grid > 1.5))[0]
            if eta_ind.shape[0] > 0:
                for ei in eta_ind:
                    if ei==0:
                        continue
                    elif petro_ratio[ei-1] > self.rpetro_eta:
                        #interpolate
                        petro_r_pixels = np.interp(self.rpetro_eta,np.flipud(petro_ratio[ei-1:ei+1]),np.flipud(radius_grid[ei-1:ei+1]))
                        delta_eta = np.abs(petro_ratio[ei-1] - petro_ratio[ei])
                        delta_r = np.abs(radius_grid[ei-1] - radius_grid[ei])
                        fsigma_eta = np.max(petro_ratio_ferr[ei-1:ei+1])
                        sigma_eta = self.rpetro_eta*fsigma_eta
                        petro_r_pixels_err = (((delta_r/delta_eta)**2)*(sigma_eta**2))**0.5  #conservative estimate of error on rpetro estimate

                        #print petro_r_pixels, delta_eta, delta_r, fsigma_eta, sigma_eta, petro_r_pixels_err, petro_ratio[ei]
                        break
                    else:
                        continue
                if petro_r_pixels<=0.0:
                    status = 'Weird light profile or too faint'
                else:
                    status = 'Positive R_pet'
            else:
                #actually it never crosses: error
                status = 'Error: problem with R_pet curve of growth'

        return petro_r_pixels, status, petro_r_pixels_err



    def init_from_synthetic_image(self,data_hdu,segmap_hdu,photutils_hdu,cm_hdu):
        self.morphtype='Synthetic Image'

        #inputs required by IDL code:
        #    morph_input_obj.write('# IMAGE  NPIX   PSF   SCALE   SKY  XC YC A/B PA SKYBOX   MAG   MAGER   DM   RPROJ[arcsec]   ZEROPT[mag?] \n')
        
        #image FITS filename 
        self.imagefile=data_hdu.header['THISFILE']
        self.image = data_hdu.data
        self.segmap = segmap_hdu.data  #general segmap containing multiple objects/labels
        self.clabel = segmap_hdu.header['CLABEL'] #label corresponding to targeted object
        #setting for doing sigma clip on internal segmap.  Not very efficient in SciPy versus IDL (why?)
        #avoid if simulated images -- not necessary if we don't expect awful pixels
        self.filter_segmap = False
        #final input for lotzmorph
        self.galaxy_segmap = np.where(self.segmap==self.clabel,self.segmap,np.zeros_like(self.segmap))
        #final image masks other objects
        self.galaxy_image = np.where(np.logical_or(self.segmap==self.clabel,self.segmap==0),self.image,np.zeros_like(self.image))
        #number of pixels in image
        self.npix = data_hdu.header['NPIX']
        xi = np.float32(np.arange(self.npix))+0.50  #center locations of pixels
        self.pixel_xpos,self.pixel_ypos = np.meshgrid(xi,xi)
        #psf in arcsec
        self.psf_fwhm_arcsec = data_hdu.header['APROXPSF']
        #scale = pixel size in arcsec
        self.pixelscale_arcsec = data_hdu.header['PIXSCALE']
        self.psf_fwhm_pixels = self.psf_fwhm_arcsec/self.pixelscale_arcsec
        #physical scale in kpc, for funsies
        self.kpc_per_arcsec = data_hdu.header['PSCALE']
        #sky = background level in image
        self.sky = data_hdu.header['SKY']
        #x and y positions. MUST CONFIRM PYTHON ORDERING/locations, 0,1 as x,y seem ok for now
        self.xcentroid = segmap_hdu.header['POS0']
        self.ycentroid = segmap_hdu.header['POS1']
        self.thisband_xcentroid = photutils_hdu.header['XCENTR']
        self.thisband_ycentroid = photutils_hdu.header['YCENTR']
        #a/b I'm guessing this is the elongation parameter?
        self.elongation = photutils_hdu.header['ELONG']
        assert (self.elongation > 0.0)
        #PA position angle.  WHAT UNITS?
        self.pa_radians = photutils_hdu.header['ORIENT'] #this looks like it's in radians, counterclockwise (photutils)
        #skybox.  do we need this if we know skysig?
        self.skysig = data_hdu.header['SKYSIG']
        #create arbitrary perfect noise image matching synthetic image properties
        #this is okay if noise is perfectly uniform gaussian right?
        self.skybox = self.skysig*np.random.randn(50,50)
        bc1 = float(self.skybox.shape[0]-1)/2.0
        bc2 = float(self.skybox.shape[1]-1)/2.0
        self.rot_skybox = skimage.transform.rotate(self.skybox,180.0,center=(bc1,bc2),mode='constant',cval=0.0,preserve_range=True)
        #AB magnitude best ... "observed" ?  aperture mags?  segment mags?
        self.magtot_intrinsic = data_hdu.header['MAG']
        self.magtot_observed = data_hdu.header['NEWMAG']  #-1 = bad
        self.magseg = photutils_hdu.header['SEGMAG'] #-1 = bad
        self.magseg_err = photutils_hdu.header['SEGMAGE'] #-1 = bad
        #distance modulus
        self.dm = data_hdu.header['DISTMOD']
        #redshift, because why not
        self.redshift = data_hdu.header['REDSHIFT']
        #rproj (arcsec)
        self.rproj_pix = photutils_hdu.header['EQ_RAD'] #pixels
        self.rproj_arcsec = self.rproj_pix*self.pixelscale_arcsec
        #AB magnitude zeropoint
        self.abzp = data_hdu.header['ABZP']

        #photutils central moments
        self.photutils_central_moments=cm_hdu.data
        #compute scale invariant moments
        scale_invariant_moments = self.compute_scale_invariant_moments()
        #compute hu translation,scale,rotation invariant moments
        hu_moments = self.compute_hu_moments()

        #Flusser & Suk 1993 Affine-invariant moments
        #nice because they are the lowest-order (less noisy) & sensitive to symmetric objects
        #note they are NOT blur invariant, but I think this is desirable (assert that PSF << features)
        fs93_moments = self.compute_fs93_moments()

        #Possible new Asymmetry indicators?: 
        #1/4 of summed magnitude of 4 3rd-order image moments
        #Should be manifestly zero for symmetric objects
        #rotation, translation, and scale invariant
        moment_asymmetry = 0.0
        magn_asym_sum = np.abs(scale_invariant_moments[3,0]) + np.abs(scale_invariant_moments[0,3]) + np.abs(scale_invariant_moments[2,1]) + np.abs(scale_invariant_moments[1,2])
        if magn_asym_sum > 0.0:
            self.m_a = np.log10(magn_asym_sum)
        else:
            self.m_a = 0.0

        #log mag of fs93 I2 affine-invariant moment (3rd-order only)
        #mag of I2
        mag_I2 = np.abs(fs93_moments[1])
        if mag_I2 > 0.0:
            self.m_I2 = np.log10(mag_I2)
        else:
            self.m_I2 = 0.0

        dummy_array = np.asarray([0.0])

        if self.morphtype != 'Synthetic Image':
            self.morph_hdu = pyfits.ImageHDU(dummy_array)
            self.morph_hdu.header['Image']=('dummy', 'What data does this image contain?')
        else:
            self.morph_hdu = pyfits.ImageHDU(self.scale_invariant_moments)
            self.morph_hdu.header['Image']=('Scale Invariant Moments', 'What data does this image contain?')

        self.morph_hdu.header['DESC']=(self.description)
        self.morph_hdu.header['TYPE']=(self.morphtype,'What kind of image was analyzed?')
        self.morph_hdu.header['Date']=(datetime.datetime.now().date().isoformat())

        return self


    
    ##work-in-progress
    def init_from_panstarrs_image(self,data_hdu,weight_hdu,segmap_hdu,se_catalog):
        self.morphtype='PanSTARRS Image'

        #se_catalog is just a single-entry ascii table after deciphering SE calc

        #inputs required by IDL code:
        #    morph_input_obj.write('# IMAGE  NPIX   PSF   SCALE   SKY  XC YC A/B PA SKYBOX   MAG   MAGER   DM   RPROJ[arcsec]   ZEROPT[mag?] \n')

        xmin = se_catalog['XMIN_IMAGE']
        xmax = se_catalog['XMAX_IMAGE']
        ymin = se_catalog['YMIN_IMAGE']
        ymax = se_catalog['YMAX_IMAGE']
        #assume object is at center with significant buffer
        xspan = xmax-xmin
        yspan = ymax-ymin
        span = np.max(np.asarray([xspan,yspan]))+50
        new_xmin = xmin - 25
        new_xmax = new_xmin + span
        new_ymin = ymin - 25
        new_ymax = new_ymin + span

        #image FITS filename 
        self.imagefile= data_hdu.fileinfo()['file'].name  #data_hdu.header['THISFILE']
        self.image = data_hdu.data[new_xmin:new_xmax,new_ymin:new_ymax]
        self.segmap = segmap_hdu.data[new_xmin:new_xmax,new_ymin:new_ymax]  #general segmap containing multiple objects/labels

        
        self.clabel = se_catalog['NUMBER'] #label corresponding to targeted object

        print np.max( self.segmap), new_xmin, new_xmax, new_ymin, new_ymax, self.imagefile, self.clabel

        
        #print self.image.shape, self.segmap.shape, self.clabel
        
        #setting for doing sigma clip on internal segmap.  Not very efficient in SciPy versus IDL (why?)
        #avoid if simulated images -- not necessary if we don't expect awful pixels
        self.filter_segmap = False
        #final input for lotzmorph
        self.galaxy_segmap = np.where(self.segmap==self.clabel,self.segmap,np.zeros_like(self.segmap))
        #final image masks other objects
        self.galaxy_image = np.where(np.logical_or(self.segmap==self.clabel,self.segmap==0),self.image,np.zeros_like(self.image))
        #number of pixels in image
        self.npix = self.image.shape[0]
        xi = np.float32(np.arange(self.npix))+0.50  #center locations of pixels
        self.pixel_xpos,self.pixel_ypos = np.meshgrid(xi,xi)
        #psf in arcsec
        self.psf_fwhm_arcsec = 1.4 #data_hdu.header['APROXPSF']
        #scale = pixel size in arcsec
        self.pixelscale_arcsec = np.abs( data_hdu.header['CD1_1']*arcsec_per_radian() )
        self.psf_fwhm_pixels = self.psf_fwhm_arcsec/self.pixelscale_arcsec
        #physical scale in kpc, for funsies
        self.kpc_per_arcsec = None #data_hdu.header['PSCALE']
        #sky = background level in image
        self.sky = 0.0 #data_hdu.header['SKY']
        #x and y positions. MUST CONFIRM PYTHON ORDERING/locations, 0,1 as x,y seem ok for now
        self.xcentroid = se_catalog['X_IMAGE'] #segmap_hdu.header['POS0']
        self.ycentroid = se_catalog['Y_IMAGE'] #segmap_hdu.header['POS1']
        self.thisband_xcentroid = self.xcentroid*1.0 #photutils_hdu.header['XCENTR']
        self.thisband_ycentroid = self.ycentroid*1.0 #photutils_hdu.header['YCENTR']
        #a/b I'm guessing this is the elongation parameter?
        self.elongation = se_catalog['ELONGATION']
        assert (self.elongation > 0.0)
        #PA position angle.  WHAT UNITS?
        self.pa_radians = se_catalog['THETA_IMAGE'] #this looks like it's in radians, counterclockwise (photutils)
        #skybox.  do we need this if we know skysig?
        self.skysig = 1.0 #data_hdu.header['SKYSIG']
        #create arbitrary perfect noise image matching synthetic image properties
        #this is okay if noise is perfectly uniform gaussian right?
        self.skybox = self.skysig*np.random.randn(50,50)
        bc1 = float(self.skybox.shape[0]-1)/2.0
        bc2 = float(self.skybox.shape[1]-1)/2.0
        self.rot_skybox = skimage.transform.rotate(self.skybox,180.0,center=(bc1,bc2),mode='constant',cval=0.0,preserve_range=True)
        #AB magnitude best ... "observed" ?  aperture mags?  segment mags?
        #self.magtot_intrinsic = data_hdu.header['MAG']
        #self.magtot_observed = data_hdu.header['NEWMAG']  #-1 = bad
        self.magseg = se_catalog['MAG_AUTO'] #-1 = bad
        self.magseg_err = se_catalog['MAGERR_AUTO'] #-1 = bad
        #distance modulus
        self.dm = None #data_hdu.header['DISTMOD']
        #redshift, because why not
        self.redshift = None #data_hdu.header['REDSHIFT']
        #rproj (arcsec)
        self.rproj_pix = 5.0 #photutils_hdu.header['EQ_RAD'] #pixels
        self.rproj_arcsec = self.rproj_pix*self.pixelscale_arcsec
        #AB magnitude zeropoint
        self.abzp = None #data_hdu.header['ABZP']

        self.m_a = 0.0
        dummy_array = np.asarray([0.0])

        self.morph_hdu = pyfits.ImageHDU(dummy_array)
        self.morph_hdu.header['Image']=('dummy', 'What data does this image contain?')

        self.morph_hdu.header['DESC']=(self.description)
        self.morph_hdu.header['TYPE']=(self.morphtype,'What kind of image was analyzed?')
        self.morph_hdu.header['Date']=(datetime.datetime.now().date().isoformat())

        return self



    
    def compute_scale_invariant_moments(self):
        scale_inv_moments = np.zeros_like(self.photutils_central_moments)
        mu_00 = self.photutils_central_moments[0,0]

        for i in np.arange(4):
            for j in np.arange(4):
                scale_inv_moments[i,j]=self.photutils_central_moments[i,j]/(mu_00**(1.0+(float(i)+float(j))/2.0))

        self.scale_invariant_moments = scale_inv_moments
        return scale_inv_moments

    def compute_fs93_moments(self):
        mu = self.photutils_central_moments
        self.fs93_I1 = (mu[2,0]*mu[0,2] - mu[1,1]**2)/(mu[0,0]**4)
        self.fs93_I2 = ((mu[3,0]**2)*(mu[0,3]**2) - 6.0*mu[3,0]*mu[2,1]*mu[1,2]*mu[0,3] + 4.0*mu[3,0]*(mu[1,2]**3) + 4.0*(mu[2,1]**3)*mu[0,3] - 3.0*(mu[2,1]**2)*(mu[1,2]**2))/(mu[0,0]**10)
        self.fs93_I3 = ( mu[2,0]*(mu[2,1]*mu[0,3] - mu[1,2]**2) - mu[1,1]*(mu[3,0]*mu[0,3]-mu[2,1]*mu[1,2]) + mu[0,2]*(mu[3,0]*mu[1,2] - mu[2,1]**2))/(mu[0,0]**7)
        self.fs93_I4 = ((mu[2,0]**3)*(mu[0,3]**2) - \
                        6.0*(mu[2,0]**2)*mu[1,1]*mu[1,2]*mu[0,3] - \
                        6.0*(mu[2,0]**2)*mu[0,2]*mu[2,1]*mu[0,3] + \
                        9.0*(mu[2,0]**2)*mu[0,2]*(mu[1,2]**2) + \
                        12.0*mu[2,0]*(mu[1,1]**2)*mu[2,1]*mu[0,3] + \
                        6.0*mu[2,0]*mu[1,1]*mu[0,2]*mu[3,0]*mu[0,3] - \
                        18.0*mu[2,0]*mu[1,1]*mu[0,2]*mu[2,1]*mu[1,2] - \
                        8.0*(mu[1,1]**3)*mu[3,0]*mu[0,3] - \
                        6.0*mu[2,0]*(mu[0,2]**2)*mu[3,0]*mu[1,2] + \
                        9.0*mu[2,0]*(mu[0,2]**2)*(mu[2,1]**2) + \
                        12.0*(mu[1,1]**2)*mu[0,2]*mu[3,0]*mu[1,2] - \
                        6.0*mu[1,1]*(mu[0,2]**2)*mu[3,0]*mu[2,1] + \
                        (mu[0,2]**3)*(mu[3,0]**2))/(mu[0,0]**11)

        self.fs93_moments = np.asarray([self.fs93_I1,self.fs93_I2,self.fs93_I3,self.fs93_I4])
        return self.fs93_moments

    def compute_hu_moments(self):
        ssim = self.scale_invariant_moments
        self.hu_I1 = ssim[2,0] + ssim[0,2]
        self.hu_I2 = (ssim[2,0]-ssim[0,2])**2 + 4.0*ssim[1,1]**2
        self.hu_I3 = (ssim[3,0]-3.0*ssim[1,2])**2 + (3.0*ssim[2,1]-ssim[0,3])**2
        self.hu_I4 = (ssim[3,0]+ssim[1,2])**2 + (ssim[2,1]+ssim[0,3])**3
        self.hu_I5 = (ssim[3,0]-3.0*ssim[1,2])*(ssim[3,0]+ssim[1,2])*((ssim[3,0]+ssim[1,2])**2 - 3.0*(ssim[2,1]+ssim[0,3])**2)+\
                     (3.0*ssim[2,1]-ssim[0,3])*(ssim[2,1]+ssim[0,3])*(3.0*(ssim[3,0]+ssim[1,2])**2-(ssim[2,1]+ssim[0,3])**2)
        self.hu_I6 = (ssim[2,0]-ssim[0,2])*( (ssim[3,0]+ssim[1,2])**2 - (ssim[2,1]+ssim[0,3])**2 ) + 4.0*(ssim[3,0]+ssim[1,2])*(ssim[2,1]+ssim[0,3])
        self.hu_I7 = (3.0*ssim[2,1]-ssim[0,3])*(ssim[3,0]+ssim[1,2])*((ssim[3,0]+ssim[1,2])**2 - 3.0*(ssim[2,1]+ssim[0,3])**2)-\
                     (ssim[3,0]-3.0*ssim[1,2])*(ssim[2,1]+ssim[0,3])*(3.0*(ssim[3,0]+ssim[1,2])**2-(ssim[2,1]+ssim[0,3])**2)
        self.hu_I8 = ssim[1,1]*((ssim[3,0]+ssim[1,2])**2 - (ssim[2,1]+ssim[0,3])**2) - (ssim[2,0]-ssim[0,2])*(ssim[3,0]+ssim[1,2])*(ssim[2,1]+ssim[0,3])
        self.hu_moments = np.asarray([self.hu_I1,self.hu_I2,self.hu_I3,self.hu_I4,self.hu_I5,self.hu_I6,self.hu_I7,self.hu_I8])

        return self.hu_moments

    def lotz_central_moments(self, i_list, j_list, xc=None, yc=None):
        new_image = np.where(self.segmap==self.clabel,self.image,np.zeros_like(self.image))

        mu_ij = central_moments(new_image,i_list,j_list,xc=xc,yc=yc)

        return np.asarray(mu_ij)

    def return_measurement_HDU(self):


        self.morph_hdu.header['Status']=(self.lotz_morph_status)
        if str.find(self.lotz_morph_status,'Error') >= 0:
            self.morph_hdu.header['FLAG']=(1,'Indicates error in morphology status')
            self.flag = 1
        else:
            self.morph_hdu.header['FLAG']=(0,'Normal completion')
            self.flag = 0

        self.morph_hdu.header['M_A'] = (self.m_a,'Log10 sum of 3rd-order scale+ invariant moments')

        if self.hu_moments[0] != None:
            self.morph_hdu.header['Hu_I1']=(self.hu_I1,'1st rotation invariant moment (Hu 1962)')
            self.morph_hdu.header['Hu_I2']=(self.hu_I2,'2nd rotation invariant moment (Hu 1962)')
            self.morph_hdu.header['Hu_I3']=(self.hu_I3,'3rd rotation invariant moment (Hu 1962)')
            self.morph_hdu.header['Hu_I4']=(self.hu_I4,'4th rotation invariant moment (Hu 1962)')
            self.morph_hdu.header['Hu_I5']=(self.hu_I5,'5th rotation invariant moment (Hu 1962)')
            self.morph_hdu.header['Hu_I6']=(self.hu_I6,'6th rotation invariant moment (Hu 1962)')
            self.morph_hdu.header['Hu_I7']=(self.hu_I7,'7th rotation invariant moment (Hu 1962)')
            self.morph_hdu.header['Hu_I8']=(self.hu_I8,'8th rotation invariant moment (Hu 1962)')

        if self.fs93_moments[0] != None:
            self.morph_hdu.header['FS93_I1']=(self.fs93_I1,'1st affine-invariant moment (Flusser & Suk 93)')
            self.morph_hdu.header['FS93_I2']=(self.fs93_I2,'2nd affine-invariant moment (Flusser & Suk 93)')
            self.morph_hdu.header['FS93_I3']=(self.fs93_I3,'3rd affine-invariant moment (Flusser & Suk 93)')
            self.morph_hdu.header['FS93_I4']=(self.fs93_I4,'4th affine-invariant moment (Flusser & Suk 93)')
            self.morph_hdu.header['M_I2']=(self.m_I2,'Log10 Magnitude of FS93_I2')

        return self.morph_hdu

    def return_rpa_segmap_hdu(self):

        if self.petro_segmap is not None:
            rpa_seg_hdu = pyfits.ImageHDU(self.petro_segmap)
            rpa_seg_hdu.header['EXTNAME']='APSEGMAP'
            return rpa_seg_hdu
        else:
            return None

    def write_idl_input_line(self,idl_filename):
        fo = open(idl_filename,'a')

        fitsfn = self.imagefile
        ababszp = self.abzp

        dm_im = self.dm
        apparent_mag = self.magseg
        absolute_mag = apparent_mag - dm_im
        me = self.magseg_err

        npix = (self.galaxy_image).shape[0]
        center = int(float(npix)/2.0)
        psfval=self.psf_fwhm_arcsec
        scaleval = self.pixelscale_arcsec

        ab = self.elongation
        #idl input in degrees, apparently, also +90deg ?
        pa = (180.0/math.pi)*self.pa_radians + 90.0 

        xc = int(self.xcentroid)
        yc = int(self.ycentroid)
        rproj = self.rproj_arcsec

        input_string = '{:80s}{:8d}{:10.3f}{:10.3f}{:10.3f}{:8d}{:8d}{:10.3f}{:7.1f}{:5.1f}{:5.1f}{:5.1f}{:5.1f}{:10.3f}{:10.3f}{:10.3f}{:10.3f}{:10.3f}\n'.format(fitsfn,npix,psfval,scaleval,0.0,xc,yc,
                                                                                                                                                             ab,pa,1.0,21.0,1.0,21.0,absolute_mag,me,dm_im,rproj,ababszp)

        fo.write(input_string)

        fo.close()
        return


    def write_py_output_line(self,python_outfile):
        fo = open(python_outfile,'a')

        fitsfn = self.imagefile
        pa = (180.0/math.pi)*self.pa_radians + 90.0 

        #Galaxy    DM   dRproj"  ABMag   ABMager  <S/N>   R(1/2)c  R(1/2)e   R_pet_c  R_pet_e  AB  PA  A_XC     A_YC     M_XC     M_YC      C    r_20    r_80    Asym    S      Gini  M_20  Flag   Cnts
        if self.morph_hdu.header['CFLAG'] != 1:
            output_string = '{:80s}{:10.3f}{:10.3f}{:10.3f}{:10.3f}'\
                            '{:10.3f}{:10.3f}{:10.3f}{:10.3f}{:10.3f}'\
                            '{:10.3f}{:10.3f}{:10.3f}{:10.3f}{:10.3f}'\
                            '{:10.3f}{:10.3f}{:10.3f}{:10.3f}{:10.3f}{:10.3f}{:10.3f}{:10.3f}{:10.3f}{:10.3f}\n'.format(fitsfn,self.dm,self.rproj_arcsec,
                                                                                                                        self.magseg,self.magseg_err,
                                                                                                                        self.snpix,0.0,0.0,self.rp_circ_2,self.rp_ellip,self.elongation,pa,
                                                                                                                        self.xcen_a2,self.ycen_a2,self.mxc,self.myc,self.cc,self.r20,self.r80,
                                                                                                                        self.asym2,0.0,self.gini,self.m20,self.flag,0.0)
        else:
            output_string = '{:80s}{:10.3f}{:10.3f}{:10.3f}{:10.3f}'\
                            '{:10.3f}{:10.3f}{:10.3f}{:10.3f}{:10.3f}'\
                            '{:10.3f}{:10.3f}{:10.3f}{:10.3f}{:10.3f}'\
                            '{:10.3f}{:10.3f}{:10.3f}{:10.3f}{:10.3f}{:10.3f}{:10.3f}{:10.3f}{:10.3f}{:10.3f}\n'.format(fitsfn,self.dm,self.rproj_arcsec,
                                                                                                                        self.magseg,self.magseg_err,
                                                                                                                        self.snpix,0.0,0.0,self.rp_circ_2,self.rp_ellip,self.elongation,pa,
                                                                                                                        self.xcen_a2,self.ycen_a2,self.mxc,self.myc,-1,-1,-1,
                                                                                                                        self.asym2,0.0,self.gini,self.m20,self.cflag,0.0)            
        fo.write(output_string)
        fo.close()
        return


#input galaxy image HDU, segmap HDU, and photutils info HDU, 
#return FITS HDU containing non-parametric morphology measurements either in the header or a FITS table HDU
def morph_from_synthetic_image(data_hdu,segmap_hdu,photutils_hdu,cm_hdu,extname='LotzMorphMeasurements',idl_filename=None,python_outfile=None,outobject=None):
    #unpack HDUs and send to generic Lotz morphology code

    galdataobject = galdata()
    galdataobject = galdataobject.init_from_synthetic_image(data_hdu,segmap_hdu,photutils_hdu,cm_hdu)


    result = galdataobject.run_lotz_morphs()
        

    morph_hdu = galdataobject.return_measurement_HDU()
    morph_hdu.header['EXTNAME']=extname

    rpa_seg_hdu = galdataobject.return_rpa_segmap_hdu()

    if idl_filename is not None and galdataobject.flag==0:
        galdataobject.write_idl_input_line(idl_filename)


    #also write output files?
    if python_outfile is not None and galdataobject.flag==0:
        galdataobject.write_py_output_line(python_outfile)

    outobject = copy.copy(galdataobject)

    return morph_hdu, rpa_seg_hdu





#work-in-progress
def morph_from_panstarrs_image(image_hdu,weight_hdu,segmap_hdu,se_catalog,extname='StatMorphMeasurements',idl_filename=None,python_outfile=None,outobject=None):

    
    galdataobject = galdata()
    galdataobject = galdataobject.init_from_panstarrs_image(image_hdu,weight_hdu,segmap_hdu,se_catalog)


    result = galdataobject.run_lotz_morphs()
        

    morph_hdu = galdataobject.return_measurement_HDU()
    morph_hdu.header['EXTNAME']=extname

    rpa_seg_hdu = galdataobject.return_rpa_segmap_hdu()

    if idl_filename is not None and galdataobject.flag==0:
        galdataobject.write_idl_input_line(idl_filename)


    #also write output files?
    if python_outfile is not None and galdataobject.flag==0:
        galdataobject.write_py_output_line(python_outfile)

    outobject = copy.copy(galdataobject)

    return morph_hdu, rpa_seg_hdu

