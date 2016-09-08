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
import scipy.ndimage
import scipy.stats as ss
import scipy.signal
import scipy as sp
import scipy.odr as odr
import glob
import os
import make_color_image
#import make_fake_wht
import gzip
import tarfile
import shutil
import cosmocalc
import congrid
import astropy.io.ascii as ascii
import sunpy__load
import sunpy__plot
import sunpy__synthetic_image
from sunpy.sunpy__plot import *
import warnings
import subprocess
import photutils
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import *
import astropy.io.fits as pyfits
import statmorph
import datetime

imdpi = 600

radii_kpc = np.asarray([0.5,1.,2.,3.,5.,10.]) #kpc
sq_arcsec_per_sr = 42545170296.0
c = 3.0e8
segment_props_save = ['area',
                      'xcentroid','ycentroid',
                      'cxx','cxy','cyy',
                      'eccentricity',
                      'ellipticity',
                      'elongation',
                      'equivalent_radius',
                      'id',
                      'max_value',
                      'maxval_xpos',
                      'maxval_ypos',
                      'min_value',
                      'minval_xpos',
                      'minval_ypos',
                      'orientation',
                      'perimeter',
                      'source_sum',
                      'semimajor_axis_sigma',
                      'semiminor_axis_sigma',
                      'xmin','xmax','ymin','ymax']

segment_props_card = ['area',
                      'xcentr','ycentr',
                      'cxx','cxy','cyy',
                      'eccent',
                      'ellipt',
                      'elong',
                      'eq_rad',
                      'id',
                      'maxvalue',
                      'maxvalx',
                      'maxvaly',
                      'min_val',
                      'minvalx',
                      'minvaly',
                      'orient',
                      'perim',
                      'seg_sum',
                      'smajsig',
                      'sminsig',
                      'xmin','xmax','ymin','ymax']


common_args = { 
                'add_background':       False,          # default option = False, turn on one-by-one
                'add_noise':            False,
                'add_psf':              True,
                'rebin_phys':           True,
                'resize_rp':            False,
                'rebin_gz':             False,           # always true, all pngs 424x424
                'scale_min':            1e-10,          # was 1e-4
                'lupton_alpha':         2e-12,          # was 1e-4
                'lupton_Q':             10,             # was ~100
                'pixelsize_arcsec':     0.25,
                'psf_fwhm_arcsec':      1.0,
                'sn_limit':             None,           # super low noise value, dominated by background
                'sky_sig':              None,           #
                'redshift':             0.05,           # 
                'b_fac':                1.1, 
                'g_fac':                1.0, 
                'r_fac':                0.9,
                'camera':               3,
                'seed_boost':           1.0,
                'save_fits':            True
                }
class analysis_parameters:
    def __init__(self,namestr=''):
        self.name = namestr

def save_photutils_props(hdu,prop):
    proptab = photutils.properties_table(prop)
    prop_array = proptab.as_array()
    for p,propstr in enumerate(segment_props_save):
        cardname = segment_props_card[p]
        hdu.header[cardname] = (prop_array[propstr][0],'Photutils segment '+propstr)
    return hdu


def test_image_mags(image_fits,ci,filtable,filter_index,redshift):
    image_hdu = pyfits.open(image_fits)[0]
    image_data = image_hdu.data
    image_header = image_hdu.header

    total_flux = np.sum(image_data)
    ababszp = image_header.get('ABZP')
    total_mag = -2.5*np.log10(total_flux)
    lum_dist = image_header.get('LUMDIST') #in Mpc
    dist_modulus = 5.0 * ( np.log10(lum_dist*1.0e6) - 1.0 )
    dist_modulus_header = image_header.get('DISTMOD')
    assert (np.abs(dist_modulus - dist_modulus_header) < 0.001)

    lambda_eff_microns = image_header.get('EFLAMBDA')

    sunrise_ab_absolute_mag = image_header.get('SUNMAG')
    image_header_ab_absolute_mag = image_header.get('ABSMAG')
    image_header_ab_apparent_mag = image_header.get('MAG')
    sunrise_independent_ab_absolute_mag = filtable.field('AB_mag_nonscatter'+str(ci))[filter_index]
    sunrise_image_apparent_mag = image_header.get('SUNAPMAG')
    sunrise_image_absolute_mag = image_header.get('SUABSMAG')

    image_ab_apparent_mag = total_mag + ababszp
    image_ab_absolute_mag = total_mag + ababszp - dist_modulus

    #10/2/2015:  Note, this will only work redward of the Ly limit, bc the integrated magnitudes in the FILTERS hdu **do not** include Lyman absorption, while images do so.
    rest_lambda_microns = lambda_eff_microns/(1.0 + redshift)

    #print rest_lambda_boundary, lambda_eff_microns, rest_lambda_microns
    if rest_lambda_microns > 0.0912 + 0.1:
        assert (np.abs(image_ab_absolute_mag - sunrise_ab_absolute_mag) < 0.2 )
        assert (np.abs(image_header_ab_absolute_mag - sunrise_ab_absolute_mag) < 0.2 )
        assert (np.abs(sunrise_ab_absolute_mag - sunrise_independent_ab_absolute_mag) < 0.01 )
        #print '   magnitudes confirmed: ', sunrise_ab_absolute_mag, image_ab_absolute_mag

    print image_ab_apparent_mag, sunrise_image_apparent_mag
    assert (np.abs(sunrise_image_apparent_mag - image_ab_apparent_mag) < 0.2 )
    assert (np.abs(sunrise_image_absolute_mag - image_ab_absolute_mag) < 0.2 )
    print '   image mags confirmed:  ', sunrise_image_apparent_mag, image_ab_apparent_mag

    #confirm lack of NANs:
    isnan = np.isnan(np.sum(image_data))
    assert (isnan==False)

    #print ababszp, total_flux, total_mag, total_mag + ababszp, dist_modulus, total_mag + ababszp - dist_modulus, sunrise_ab_absolute_mag

    return 1


def initialize_test_figure():
    fig = pyplot.figure(figsize=(5.0,4.0), dpi=imdpi)
    pyplot.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0,wspace=0.0,hspace=0.0)
    nx = 5.0
    ny = 4.0

    return fig,1.0/nx,1.0/ny,int(nx),int(ny)


def generate_filter_images(bbfile, snapnum,subdirnum,sh_id,ci,custom_filename_sb00,analysis_object,i,clobber=False,analyze=True,openlist=None,snprefix=None):
    filter_index = analysis_object.filter_indices[i]
    filter_label = analysis_object.filter_labels[i]
    psf_file = analysis_object.psf_files[i]
    psf_pix_arcsec = analysis_object.psf_pix_arcsec[i]
    psf_truncate = analysis_object.psf_truncate[i]
    psf_hdu = analysis_object.psf_hdu_num[i]
    photfnu_Jy_i = analysis_object.photfnu_Jy[i]

    common_args['pixelsize_arcsec'] = analysis_object.pixsize_arcsec[i]

    print filter_index, filter_label, i, type(filter_index)


    if not os.path.lexists(custom_filename_sb00) or clobber==True:
        #do sunpy calcs
        print custom_filename_sb00, filter_index, filter_label, psf_file, psf_pix_arcsec, psf_hdu, psf_truncate
        cam_0_raw, rp, the_used_seed,this_fail_flag,fitsfn,openlist   = sunpy__synthetic_image.build_synthetic_image(bbfile, filter_index,
                                                                                                            seed=0,
                                                                                                            r_petro_kpc=None, 
                                                                                                            fix_seed=False,
                                                                                                            custom_fitsfile=custom_filename_sb00,
                                                                                                            psf_fits=psf_file,
                                                                                                            psf_pixsize_arcsec=psf_pix_arcsec,
                                                                                                            psf_truncate_pixels = psf_truncate,
                                                                                                            psf_hdu_num = psf_hdu,
                                                                                                            openlist=openlist,
                                                                                                            **common_args)
        assert (os.path.lexists(custom_filename_sb00))

        #add some useful info to header
        #data,header = pyfits.getdata(custom_filename_sb00, ext=0, header=True)
        hdulist = pyfits.open(custom_filename_sb00)
        if snapnum is not None:
            hdulist[0].header['SNAPNUM']=(snapnum,'Illustris snapshot number')
            hdulist[0].header['SUBDIR']=(subdirnum,'Image subdirectory')
            hdulist[0].header['SUBH_ID']=(sh_id,'Subhalo ID index')
            hdulist[0].header['REF']=('Torrey et al. (2015)', 'ideal image reference')
        else:
            hdulist[0].header['REF']=('Moody+ 2013; Snyder+ 2015a', 'ideal image reference')
            hdulist[0].header['SNPREFIX']=(snprefix, 'VELA snap ID')


        hdulist[0].header['CAMERA']=(ci,'Sunrise camera number')
        hdulist[0].header['FLABEL']=(filter_label,'Filter shorthand')
        hdulist[0].header['HSTREF']=('HST-AR#13887 (PI G Snyder)','realistic image reference')
        hdulist[0].header['Date']=(datetime.datetime.now().date().isoformat())
        hdulist[0].header['PHOTFNU']=(photfnu_Jy_i,'Jy; inverse sensitivity, flux[Jy] giving 1 count/sec')
        hdulist[0].header['APROXPSF']=(analysis_object.psf_fwhm_arcsec[i],'Estimate of PSF FWHM in arcsec')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            #pyfits.writeto(custom_filename_sb00, data, header, clobber=True)
            hdulist.writeto(custom_filename_sb00,clobber=True)

    else:
        print 'SB00 file exists, skipping: ', custom_filename_sb00

        #after if statement, file must exist
        assert (os.path.lexists(custom_filename_sb00))

        #confirm magnitudes make sense
        #only for testing
        #result = test_image_mags(custom_filename_sb00,ci,filtable,filter_index,redshift)

    #Do noise additions
    for mag_i,maglim in enumerate(analysis_object.magsb_limits):
        custom_filename = custom_filename_sb00.rstrip('SB00.fits')+'SB{:2.0f}.fits'.format(maglim)
        if not os.path.lexists(custom_filename) or clobber==True:
            ### is this the best one???
            #print maglim
            sigma_nJy = (2.0**(-0.5))*((1.0e9)*(3631.0/5.0)*10.0**(-0.4*maglim))*analysis_object.pixsize_arcsec[i]*(3.0*analysis_object.psf_fwhm_arcsec[i])
            sigma_muJyAs = (sigma_nJy*1.0e-3)/(analysis_object.pixsize_arcsec[i]**2)
            noiseless_hdu = pyfits.open(custom_filename_sb00)[0]
            noiseless_image = noiseless_hdu.data
            noiseless_header = noiseless_hdu.header
            npix = noiseless_image.shape[0]
            noise_image = sigma_muJyAs*np.random.randn(npix,npix)
            #assumes t -> infinity, effective_gain -> infinity, i.e. very long integration times
            
            new_image = noiseless_image+noise_image
            
            primhdu = pyfits.PrimaryHDU(np.float32(new_image),header=noiseless_header)
            primhdu.header['SBMAGLIM']=(round(maglim,6),'mag/SqArcsec')
            primhdu.header['RMS']=(round(sigma_muJyAs,6),'muJy/SqArcsec')
            primhdu.header['SKYSIG']=(round(sigma_muJyAs,6),'image units')
            primhdu.header['SKY']=(round(0.0,6),'image units')
            newflux = np.sum(noiseless_image+noise_image)
            newmag = -1
            if newflux > 0.0:
                newmag = -2.5*np.log10(newflux) + noiseless_header.get('ABZP')
                        
            #print newmag, noiseless_header.get('ABZP'), np.sum(noiseless_image+noise_image), np.sum(noiseless_image), sigma_nJy, sigma_muJyAs
                        
            primhdu.header['NEWMAG']=(round(newmag,6),'includes noise, -1 bad')
                        
            #nhdu = pyfits.ImageHDU(np.float32(noise_image))
            #nhdu.update_ext_name('NOISE')
                        
            hdulist = pyfits.HDUList([primhdu])
                        
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                #save container to file, overwriting as needed
                hdulist.writeto(custom_filename,clobber=clobber)
        else:
            print 'SB?? file exists, skipping: ', custom_filename

        #do segmentation and initial photometry
        #always do this-- it's fast enough
        if True:
            #if (filter_label=='WFC3-F160W') or (filter_label=='NC-F200W') or (filter_label=='NC-F444W'):
            #switch to one-per-redshift for great simplicity
            if filter_label==analysis_object.segment_filter_label:

                print '   segmenting '+filter_label, maglim, custom_filename

                res = segment_image(custom_filename,filter_label = filter_label)
                #do initial plotting/saving inside this routine?
                


    return openlist


def analyze_image_morphology(custom_filename,filter_index,segmap_filename,segmap_hdu,
                             seg_image,seg_header,seg_npix,seg_filter_label,clabel,cpos0,cpos1,
                             figure,nx,ny,totalcount,analyze=True,clobber=False,idl_filename=None,python_outfile=None):

    assert (os.path.lexists(custom_filename))

    if analyze==True:

        #open existing HDU list
        existing_hdulist = pyfits.open(custom_filename,memmap=True)
        image_hdu = existing_hdulist['SYNTHETIC_IMAGE']
        data = image_hdu.data
        rms = image_hdu.header['RMS']
        pix_arcsec = image_hdu.header['PIXSCALE']
        pscale = image_hdu.header['PSCALE']
        typical_kpc_per_arcsec = 8.0 #image_hdu.header['PSCALE']
        abzp = image_hdu.header['ABZP']
        npix = image_hdu.header['NPIX']

        image_hdu.header['THISFILE']=(custom_filename)
        image_hdu.header['SEGFILE']=(segmap_filename)

        #perform aperture photometry based on relevant filter segmentation... one per redshift
        radii_arcsec = radii_kpc/pscale
        radii_pixels = radii_arcsec/pix_arcsec
                    
        #for simplicity, clobber old file, otherwise this becomes nuts
        new_hdulist = pyfits.HDUList([image_hdu])

        #polish segmap
        if seg_npix != npix:
            rebinned_segmap = np.int16( congrid.congrid(seg_image, (npix,npix), method='neighbour', centre=True, minusone=False) )
            #print rebinned_segmap.shape, np.min(rebinned_segmap), np.max(rebinned_segmap)
            saveseg_hdu = pyfits.ImageHDU(rebinned_segmap,header=seg_header)
            #also must update position data
            position_ratio = float(npix)/float(seg_npix)
            new_cpos0 = cpos0*position_ratio
            new_cpos1 = cpos1*position_ratio
            saveseg_hdu.header['POS0']=new_cpos0
            saveseg_hdu.header['POS1']=new_cpos1
            #must confirm validity of centering data
        else:
            rebinned_segmap = seg_image
            #print rebinned_segmap.shape, np.min(rebinned_segmap), np.max(rebinned_segmap)
            saveseg_hdu = segmap_hdu


        #save segmap
        new_hdulist.append(saveseg_hdu)


        apertures = []
        #run segment properties on new thing
        if clabel != 0:
            new_props = photutils.source_properties(data, rebinned_segmap)
            nc = 0
            for pi,prop in enumerate(new_props):
                if prop.id==clabel:
                    nc = nc+1
                    center_prop = prop
                    #print center_proptab
                    position = (center_prop.xcentroid.value, center_prop.ycentroid.value)
                    position_fixed = (saveseg_hdu.header['POS0'],saveseg_hdu.header['POS1'])
                    r=3
                    a = center_prop.semimajor_axis_sigma.value * r
                    b = center_prop.semiminor_axis_sigma.value * r
                    theta = center_prop.orientation.value
                    aperture = photutils.EllipticalAperture(position, a, b, theta=theta)
                    apertures.append(aperture)
                    
                    tbhdu = do_aperture_photometry(data,radii_pixels,radii_kpc,radii_arcsec,position_fixed,rms,abzp,extname='PhotUtilsMeasurements')
                    tbhdu.header['a']=(a,'isophotal semimajor axis sigma from photutils')
                    tbhdu.header['b']=(b,'isophotal semiminor axis sigma from photutils')
                    tbhdu.header['r']=(r,'isophotal radius assumed for semi-axes in photutils')
                    tbhdu.header['theta']=(theta,'photutils orientation parameter')
                    tbhdu = save_photutils_props(tbhdu,center_prop)
                    
                    segment_npix = tbhdu.header['AREA']
                    segment_flux = tbhdu.header['SEG_SUM']
                    
                    segment_fluxerr = 0
                    segment_mag = -1
                    segment_magerr = -1
                    if segment_flux > 0:
                        segment_fluxerr = (segment_npix*rms**2)**0.5
                        segment_mag = -2.5*np.log10(segment_flux) + abzp
                        segment_magerr = 2.5*np.log10(np.exp(1))*(segment_fluxerr/segment_flux)

                    tbhdu.header['SEGFERR']=(segment_fluxerr,'segment flux error in image units')
                    tbhdu.header['SEGMAG']=(segment_mag,'segment AB magnitude, -1=bad')
                    tbhdu.header['SEGMAGE']=(segment_magerr,'error on segment AB magnitude, -1=bad')

                    new_hdulist.append(tbhdu)

                    cmhdu = pyfits.ImageHDU(center_prop.moments_central)
                    cmhdu.header['EXTNAME']=('PhotUtilsCentralMoments')
                    new_hdulist.append(cmhdu)

                    if segment_magerr >= 0.0 and segment_magerr < 0.4:

                        print '        '
                        print '        -----------------------------------------------------------------------'
                        print '        Measuring Morphologies of: ', custom_filename
                        print '        Mag: {:5.2f}    Magerr:  {:5.2f}'.format(segment_mag,segment_magerr)
                        try:
                            mhdu, ap_seghdu = lotzmorph.morph_from_synthetic_image(image_hdu,saveseg_hdu,tbhdu,cmhdu,extname='LotzMorphMeasurements',idl_filename=idl_filename,python_outfile=python_outfile)
                        except:
                            print "Exception inside morphology analysis code! ", custom_filename
                            mhdu = None
                            ap_seghdu = None
                        else:
                            new_hdulist.append(mhdu)
                            if ap_seghdu is not None:
                                new_hdulist.append(ap_seghdu)
                            
                        print '        '

                    else:
                        print '       '
                        print '        -----------------------------------------------------------------------'
                        print '        Skipping Morphologies of: ', custom_filename
                        print '        Mag: {:5.2f}    Magerr:  {:5.2f}'.format(segment_mag,segment_magerr)
                        print '       '
                        mhdu = None
                        ap_seghdu = None

                    result = plot_test_stamp(image_hdu,saveseg_hdu,tbhdu,cmhdu,mhdu,ap_seghdu,figure,nx,ny,totalcount)

                    assert (nc<=1)
                        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            #save container to file, overwriting as needed
            new_hdulist.writeto(custom_filename,clobber=True)


    return 1

def plot_test_stamp(image_hdu,saveseg_hdu,tbhdu,cmhdu,mhdu,ap_seghdu,figure,nx,ny,totalcount):
    #initialize axis
    axi = figure.add_subplot(ny,nx,totalcount+1) 
    axi.set_xticks([]) ; axi.set_yticks([])

    #plot grayscale galaxy image
    data = image_hdu.data
    norm = ImageNormalize(stretch=LogStretch(),vmin=0.9*image_hdu.header['RMS'],vmax=np.max(data),clip=True)
    axi.imshow(data, origin='lower', cmap='Greys_r', norm=norm, interpolation='nearest')
    axi.annotate('{:3.2f}$\mu m$'.format(image_hdu.header['EFLAMBDA']),xy=(0.05,0.05),xycoords='axes fraction',color='white',ha='left',va='center',size=6)

    #plot initial photutils segmentation map contour
    segmap = saveseg_hdu.data
    clabel = saveseg_hdu.header['CLABEL']
    segmap_masked = np.where(segmap==clabel,segmap,np.zeros_like(segmap))
    axi.contour(segmap_masked, (clabel-0.0001,), linewidths=0.1, colors=('DodgerBlue',))


    centrsize = 1
    #plot centroid
    axi.plot([tbhdu.header['POS0']],[tbhdu.header['POS1']],'o',color='DodgerBlue',markersize=centrsize,alpha=0.6,mew=0)
    axi.plot([tbhdu.header['XCENTR']],[tbhdu.header['YCENTR']],'o',color='Yellow',markersize=centrsize,alpha=0.6,mew=0)

    #plot asymmetry center and elliptical (and circular) petrosian radii ???
    if mhdu is not None:
        if mhdu.header['FLAG']==0:
            axc = mhdu.header['AXC']
            ayc = mhdu.header['AYC']
            rpe = mhdu.header['RPE']
            elongation = mhdu.header['ELONG']
            position = (axc, ayc)
            a = rpe
            b = rpe/elongation
            theta = mhdu.header['ORIENT']
            aperture = photutils.EllipticalAperture(position, a, b, theta=theta)
            aperture.plot(color='Orange', alpha=0.4, ax=axi,linewidth=1)
            axi.plot([axc],[ayc],'+',color='Orange',markersize=centrsize,mew=0.1)


    #plot petrosian morphology segmap in different linestyle
    if ap_seghdu is not None:
        ap_segmap = ap_seghdu.data
        axi.contour(ap_segmap, (10.0-0.0001,), linewidths=0.1,colors='Orange')

    return 1

def process_single_broadband(bbfile,analysis_object,bbase='broadband_red_',clobber=False, analyze=True, do_idl=False):
    #create subdirectory to hold mock images and analyses
    end = bbfile[-3:]
    if end=='.gz':
        is_unzipped=False
    else:
        is_unzipped=True

    if bbase is not "broadbandz":
        if is_unzipped:
            sh_id = bbfile[len(bbase):].rstrip('.fits')
        else:
            sh_id = bbfile[len(bbase):].rstrip('.fits.gz')
            
        bb_dir = 'images_subhalo_'+sh_id
        subdir_path = os.path.dirname(os.path.abspath(bbfile))
        subdirnum = subdir_path[-3:]
        snap_path = os.path.dirname(subdir_path)
        snapnum = snap_path[-3:]
        snap_prefix = 'snap'+snapnum+'dir'+subdirnum+'sh'+sh_id
    else:
        snapnum=None
        subdirnum=None
        sh_id=None
        snap_prefix = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(bbfile))))
        bb_dir = 'images_'+snap_prefix

    print bb_dir
    if not os.path.lexists(bb_dir):
        os.mkdir(bb_dir)



    #calculate number of cameras
    assert (analysis_object.camera_indices is not None)
    use_camind = analysis_object.camera_indices


    all_files_exist = True

    #check if file is gzipped and if we need to access the data at all
    for camindex,ci in enumerate(use_camind):
        camstring = '{:02}'.format(ci)
        common_args['camera']=ci

        for i,filter_label in enumerate(analysis_object.filter_labels):
            #naming convention
            #snap???dir???sh*cam??_[FILTER]_SB??.fits
            #SB00 noiseless
            custom_filename_sb00 = os.path.join(bb_dir,snap_prefix+'cam'+camstring+'_'+filter_label+'_SB00.fits')
            if not os.path.lexists(custom_filename_sb00):
                all_files_exist=False
    
    print "All files exist? ", all_files_exist

    if (not is_unzipped and not all_files_exist) or (not is_unzipped and clobber is True):
        subprocess.call(['gunzip', bbfile])
        bbfile = bbfile.rstrip('.gz')
        is_unzipped = True

    bb_hdulist = pyfits.open(bbfile,memmap=False)
    common_args['redshift']=bb_hdulist['BROADBAND'].header['REDSHIFT']

    openlist = None

    #Step 1 -- loop over filters (PSF) and depths to create mock images
    for camindex,ci in enumerate(use_camind):
        camstring = '{:02}'.format(ci)
        common_args['camera']=ci

        for i,filter_label in enumerate(analysis_object.filter_labels):
            sys.stdout.flush()
            #naming convention
            #snap???dir???sh*cam??_[FILTER]_SB??.fits
            #SB00 noiseless
            custom_filename_sb00 = os.path.join(bb_dir,snap_prefix+'cam'+camstring+'_'+filter_label+'_SB00.fits')


            try:
                openlist = generate_filter_images(bbfile,snapnum,subdirnum,sh_id,ci,custom_filename_sb00, analysis_object, i, clobber=clobber,analyze=analyze,openlist=openlist,snprefix=snap_prefix)
            except (KeyboardInterrupt,NameError,AttributeError,TypeError,IndexError) as e:
                print e
                raise
            except:
                print "Exception while processing filter image creation: ", filter_label, custom_filename_sb00
                print "Error:", sys.exc_info()[0]


    if openlist is not None:
        openlist.close()
    

    #compress when finished with broadband.fits file
    if is_unzipped is True:
        subprocess.call(['gzip', '-9', bbfile])


    #morphology code requires existence of useful segmentation maps
    #therefore, need a new loop to measure images after deciding on segmentation
    #inside this loop, do:
    #1.  aperture photometry and source morphology with photutils, for each filter & one segmap
    #2.  lotz++ morphologies

    #write a Lotz IDL input file for comparison tests
    idl_input_file = os.path.join(bb_dir,snap_prefix+'_idlinput.txt')
    idl_obj = open(idl_input_file,'w')
    idl_obj.write('# IMAGE  NPIX   PSF   SCALE   SKY  XC YC A/B PA SKYBOX   MAG   MAGER   DM   RPROJ[arcsec]   ZEROPT[mag?] \n')
    idl_obj.close()
    #    morph_input_obj.write('# IMAGE  NPIX   PSF   SCALE   SKY  XC YC A/B PA SKYBOX   MAG   MAGER   DM   RPROJ[arcsec]   ZEROPT[mag?] \n')
    idl_output_file = os.path.join(bb_dir,snap_prefix+'_idloutput.txt')
    py_output_file = os.path.join(bb_dir,snap_prefix+'_pyoutput.txt')
    if os.path.lexists(py_output_file):
        os.remove(py_output_file)

    
    for camindex,ci in enumerate(use_camind):
        camstring = '{:02}'.format(ci)

        for mag_i,maglim in enumerate(analysis_object.magsb_limits):

            #segmap is well defined now, find and load it here
            segmap_filename_sb00 = os.path.join(bb_dir,snap_prefix+'cam'+camstring+'_'+analysis_object.segment_filter_label+'_SB00.fits')
            segmap_filename = segmap_filename_sb00.rstrip('SB00.fits')+'SB{:2.0f}.fits'.format(maglim)
            segmap_hdu = pyfits.open(segmap_filename)['SEGMAP']
            seg_image = segmap_hdu.data
            seg_header = segmap_hdu.header
            seg_npix = seg_image.shape[0]
            seg_filter_label = seg_header['FILTER']
            clabel = segmap_hdu.header['CLABEL']
            cpos0 = segmap_hdu.header['POS0']
            cpos1 = segmap_hdu.header['POS1']
            print '   loaded segmentation map with properties ', segmap_filename, seg_npix, clabel, cpos0, cpos1

            #one figure per depth and viewing angle -- all filters
            outfigname = os.path.join(bb_dir,snap_prefix+'cam'+camstring+'_'+'SB{:2.0f}'.format(maglim)+'_test.pdf')
            figure,deltax,deltay,nx,ny = initialize_test_figure()
            axiscount = 0


            for i,filter_label in enumerate(analysis_object.filter_labels):
                sys.stdout.flush()

                filter_index = analysis_object.filter_indices[i]
                filter_lambda_order = analysis_object.filter_lambda_order[i]
                custom_filename_sb00 = os.path.join(bb_dir,snap_prefix+'cam'+camstring+'_'+filter_label+'_SB00.fits')
                custom_filename = custom_filename_sb00.rstrip('SB00.fits')+'SB{:2.0f}.fits'.format(maglim)
                skipfilter = analysis_object.skip_filters[i]

                if skipfilter or not analyze:
                    analyze_this=False
                else:
                    analyze_this=True

                try:
                    result = analyze_image_morphology(custom_filename,filter_index,segmap_filename,segmap_hdu,
                                                      seg_image,seg_header,seg_npix,seg_filter_label,clabel,cpos0,cpos1,
                                                      figure,nx,ny,filter_lambda_order,analyze=analyze_this,clobber=clobber,
                                                      idl_filename=idl_input_file,python_outfile=py_output_file)
                    axiscount=axiscount+1
                except (KeyboardInterrupt, AttributeError) as e:
                    print e
                    raise
                except:
                    print "Exception while analyzing image morphology: ", custom_filename
                    print "Error:", sys.exc_info()[0]
                else:
                    pass

            #save test figure, one for each depth
            figure.savefig(outfigname,dpi=imdpi)
            pyplot.close(figure)



    #run IDL morph for comparison tests
    #pathnames relative to bbdir for quicker running in a high-level loop

    runscript = os.path.join(bb_dir,'run_morph_script.sh')
    ro = open(runscript,'w')
    ro.write('export IDL_PATH=$IDL_PATH:$HOME/Dropbox/Projects/IDL_code/pro/morphs_illustris_comparison/idlpros\n')
    ro.write('idl '+bb_dir+'/morphscript.pro > '+bb_dir+'/run_morph.out 2> '+bb_dir+'/run_morph.err\n')
    ro.close()
    
    morphscript = os.path.join(bb_dir,'morphscript.pro')
    mo = open(morphscript,'w')
    mo.write('domorphs_gfs, "'+idl_input_file+'", "'+idl_output_file+'", 0, start=0, finish=1000\n')
    mo.write('exit\n')
    mo.close()


    if do_idl:
        if os.path.lexists(idl_output_file):
            os.remove(idl_output_file)
        print '        -----------------------------------------------------------------------'
        print '        Measuring Morphologies with IDL Code... '
        subprocess.call(['bash', runscript])

    return 1





def segment_image(filename,filter_label='None'):
    existing_hdulist = pyfits.open(filename,mode='append',memmap=True) 

    segmap_exists=False
    if len(existing_hdulist)>1:
        segmap_exists=True
        return 2
        #might as well skip this?

    image_hdu = existing_hdulist['SYNTHETIC_IMAGE']
    data = image_hdu.data
    rms = image_hdu.header['RMS']
    pix_arcsec = image_hdu.header['PIXSCALE']
    #images are officially background-subtracted

    if segmap_exists:
        return existing_hdulist['SEGMAP'],data

    thr_rms = 1.3
    thresh = thr_rms * rms
    npixels = 5

    #build kernel for pre-filtering.  How big?
    #if we use PSCALE, we're pre-supposing redshift information
    pscale = image_hdu.header['PSCALE']
    typical_kpc_per_arcsec = 8.0 #image_hdu.header['PSCALE']

    abzp = image_hdu.header['ABZP']

    kernel_kpc_fwhm = 2.0
    kernel_arcsec_fwhm = kernel_kpc_fwhm/typical_kpc_per_arcsec
    kernel_pixel_fwhm = kernel_arcsec_fwhm/pix_arcsec
    #print '   segmentation kernel fwhm kpc, arcsec, pix ',kernel_kpc_fwhm,kernel_arcsec_fwhm,kernel_pixel_fwhm

    sigma = kernel_pixel_fwhm * gaussian_fwhm_to_sigma
    nsize = int(5*kernel_pixel_fwhm)
    kernel = Gaussian2DKernel(sigma, x_size=nsize, y_size=nsize)

    segmap_obj = photutils.detect_sources(data, thresh, npixels=npixels, filter_kernel=kernel)
    segmap = segmap_obj.data
    props = photutils.source_properties(data, segmap)
    #save segmap and info
    nhdu = pyfits.ImageHDU(np.int16(segmap)) #limits to 256 objects
    nhdu.header['EXTNAME']=('SEGMAP')
    nhdu.header['BKG'] = (0.0,'Image background')
    nhdu.header['RMS'] = (rms,'Image background RMS')
    nhdu.header['THR_RMS'] = (round(thr_rms,6),'Factor of RMS for extraction')
    nhdu.header['THR_IM'] = (round(thresh,6),'Threshold value for extraction')
    nhdu.header['NPIXELS'] = (round(npixels,6),'Minimum number of pixels')
    nhdu.header['KFWHM'] = (round(kernel_pixel_fwhm,6),'Kernel FWHM in pixels')
    nhdu.header['KFWHMKPC'] = (round(kernel_kpc_fwhm,6),'Kernel FWHM in estimated kpc')
    nhdu.header['KFWHMAS'] = (round(kernel_arcsec_fwhm,6),'Kernel FWHM in arcsec')



    #ID target galaxy and mask rest?
    N = data.shape[0]
    Nc = 1
    checki = segmap[N/2-Nc:N/2+Nc,N/2-Nc:N/2+Nc]  #center 4 or 9 pixels
    values = np.unique(checki)
    values_nonzero = np.where(values > 0)[0]

    #print N, Nc, N/2, N/2-Nc, N/2 + Nc
    #print checki
    #print values
    #print values_nonzero, values_nonzero.shape

    if values_nonzero.shape[0] ==1:
        center_label = values[values_nonzero[0]]
        print '   found center label unanimously', center_label, filename
    elif values_nonzero.shape[0] > 1:
        sums = props[values[values_nonzero[0]]-1].source_sum
        center_label = np.argmax(sums) + 1
        print '   found center label by combat', center_label, filename, sums, values_nonzero, values[values_nonzero]
    else:
        center_label = 0
        print '   unable to locate central source', values, filename

    max_label = np.max(segmap)


    cpos0 = 0.0
    cpos1 = 0.0

    apertures = []
    if max_label > 0:   
        r = 3.    # approximate isophotal extent
        colors = ['DodgerBlue','Yellow','ForestGreen','GoldenRod','Pink','Red','White','Turquoise']

        for oi,prop in enumerate(props):
            position = (prop.xcentroid.value, prop.ycentroid.value)
            a = prop.semimajor_axis_sigma.value * r
            b = prop.semiminor_axis_sigma.value * r
            theta = prop.orientation.value
            aperture = photutils.EllipticalAperture(position, a, b, theta=theta)
            apertures.append(aperture)

            label = prop.id
            #segmap_masked = np.where(segmap==label,segmap,np.zeros_like(segmap))
            colori = min(label-1,7)
            #axi.contour(segmap_masked, (label-0.0001,), linewidths=0.3, colors=colors[colori])

        
            if center_label != None:
                if label==center_label:
                    #aperture.plot(color=colors[colori], alpha=1.0, ax=axi,linestyle='dashed',linewidth=0.5)
                    cpos0 = position[0]
                    cpos1 = position[1]


    nhdu.header['CLABEL']=(center_label,'Segmap label of targeted object')
    nhdu.header['POS0']=(cpos0,'0 position of targeted object')
    nhdu.header['POS1']=(cpos1,'1 position of targeted object')
    nhdu.header['FILTER']=(filter_label,'detection and segmentation on this filter')
    #hdulist = pyfits.HDUList([image_hdu,nhdu])


    if segmap_exists:
        existing_hdulist.close()
        pyfits.update(filename,nhdu.data,nhdu.header,'SEGMAP')
    else:
        existing_hdulist.append(nhdu)
        existing_hdulist.flush()
        existing_hdulist.close()


    #with warnings.catch_warnings():
    #    warnings.simplefilter("ignore")
    #    #save container to file, overwriting as needed
    #    hdulist.writeto(filename,clobber=True)

    return 1


def do_aperture_photometry(data,radii_pixels,radii_kpc,radii_arcsec,position,rms,abzp,extname='AperturePhotometry'):

    flux = []
    flux_err = []
    mag = []
    mag_err = []

    for radius in radii_pixels:
        phot_table = photutils.aperture_photometry(data, photutils.CircularAperture(position, radius), error=rms, pixelwise_error=False, effective_gain = None)
        flux.append(phot_table['aperture_sum'])
        flux_err.append(phot_table['aperture_sum_err'])
        if phot_table['aperture_sum'] > 0.0:
            mag.append( -2.5*np.log10(phot_table['aperture_sum'].data[0]) + abzp )
            mag_err.append(1.0857*phot_table['aperture_sum_err'].data[0]/phot_table['aperture_sum'].data[0])
        else:
            mag.append(-1.0)
            mag_err.append(-1.0)
    

    tbhdu = pyfits.BinTableHDU.from_columns([pyfits.Column(name='radii_kpc', format='F', array=radii_kpc),
                                                             pyfits.Column(name='radii_arcsec', format='F', array=radii_arcsec),
                                                             pyfits.Column(name='radii_pixels', format='F', array=radii_pixels),
                                                             pyfits.Column(name='flux', format='F', array=np.asarray(flux)),
                                                             pyfits.Column(name='flux_err', format='F', array=np.asarray(flux_err)),
                                                             pyfits.Column(name='mag', format='F', array=np.asarray(mag)),
                                                             pyfits.Column(name='mag_err', format='F', array=np.asarray(mag_err))])

    tbhdu.header['EXTNAME']=(extname)
    tbhdu.header['ERRORS']=('Sky Only', 'Errors neglect source-based Poisson noise')
    tbhdu.header['POS0']=(position[0],'0 position for aperture photometry')
    tbhdu.header['POS1']=(position[1],'1 position for aperture photometry')

    return tbhdu

def process_subdir(subdirpath='.',mockimage_parameters=None,clobber=False, max=None, galaxy=None,seg_filter_label='NC-F200W',magsb_limits=[23.0,25.0,27.0,29.0],camindices=[0,1,2,3],do_idl=False,analyze=True):
    cwd = os.path.abspath(os.curdir)

    os.chdir(subdirpath)

    bbfile_list = np.sort(np.asarray(glob.glob('broadband_red_*.fits*')))   #enable reading .fits.gz files
    print bbfile_list

    if galaxy is not None:
        thisbb = np.where(bbfile_list==galaxy)[0]
        bbfile_list= bbfile_list[thisbb]

    test_file = bbfile_list[0]
    tf = pyfits.open(test_file)
    print tf.info()
    print tf['BROADBAND'].header.cards
    print tf['SFRHIST'].header.get('star_adaptive_smoothing')
    print tf['SFRHIST'].header.get('star_radius_factor')

    #this is critical for later
    
    fils = tf['FILTERS'].data.field('filter')
    print fils


    #morph_input_file = 'morph_input_gSDSS_starsizetests_cam'+str(common_args['camera'])+'.txt'
    #print morph_input_file
    #morph_input_obj = open(morph_input_file,'w')
    #morph_input_obj.write('# IMAGE  NPIX   PSF   SCALE   SKY  XC YC A/B PA SKYBOX   MAG   MAGER   DM   RPROJ[arcsec]   ZEROPT[mag?] \n')


    filters_to_analyze = ['ACS_F435_NEW.res',
                          'ACS_F606_NEW.res',
                          'ACS_F775_NEW.res',
                          'ACS_F850_NEW.res',
                          'f105w.IR.res',
                          'f125w.IR.res',
                          'f160w.IR.res',
                          'NIRCAM_prelimfiltersonly_F070W',  #names too long for filter table
                          'NIRCAM_prelimfiltersonly_F090W',
                          'NIRCAM_prelimfiltersonly_F115W',
                          'NIRCAM_prelimfiltersonly_F150W',
                          'NIRCAM_prelimfiltersonly_F200W',
                          'NIRCAM_prelimfiltersonly_F277W',
                          'NIRCAM_prelimfiltersonly_F356W',
                          'NIRCAM_prelimfiltersonly_F444W',
                          'f140w.IR.res', 
                          'f275w.UVIS1.res',
                          'f336w.UVIS1.res',
                          'F814W_WFC.res']

    skip_filter_boolean = [False,
                           False,
                           True,
                           True,
                           False,
                           True,
                           False,
                           True,
                           True,
                           False,
                           False,
                           False,
                           False,
                           False,
                           False,
                           True,
                           True,
                           False,
                           False]


    print filters_to_analyze
    
    pixsize_arcsec = [0.03,0.03,0.03,0.03,0.06,0.06,0.06,0.032,0.032,0.032,0.032,0.032,0.065,0.065,0.065,0.06,0.03,0.03,0.03]

    filter_labels = ['ACS-F435W','ACS-F606W','ACS-F775W','ACS-F850LP','WFC3-F105W','WFC3-F125W','WFC3-F160W',
                     'NC-F070W','NC-F090W','NC-F115W','NC-F150W','NC-F200W','NC-F277W','NC-F356W','NC-F444W',
                     'WFC3-F140W','WFC3-F275W','WFC3-F336W','ACS-F814W']

    filter_indices = []

    for i,f in enumerate(filters_to_analyze):
        fi = np.where(fils==f)
        print fi[0][0], f, fils[fi[0][0]], filter_labels[i] #, filters_to_analyze[fi]
        filter_indices.append(fi[0][0])


    filter_indices = np.asarray(filter_indices)

    print filter_indices

    filter_lambda_order = [2,3,4,6,7,8,10,
                           11,12,13,14,15,16,17,18,
                           9,0,1,5]


    #photfnu units Jy; flux in 1 ct/s
    photfnu_Jy = [1.96e-7,9.17e-8,1.97e-7,4.14e-7,
                  1.13e-7,1.17e-7,1.52e-7,
                  5.09e-8,3.72e-8,3.17e-8,2.68e-8,2.64e-8,2.25e-8,2.57e-8,2.55e-8,
                  9.52e-8,8.08e-7,4.93e-7,1.52e-7]

    morphcode_dir = "/Users/gsnyder/Documents/pro/morph_december2013/morph_pro/"
    morphcode_files = np.asarray(glob.glob(os.path.join(morphcode_dir,"*.*")))

    se_dir = '/Users/gsnyder/Documents/Projects/Illustris_Morphology/Illustris-CANDELS/SE_scripts'
    se_files = np.asarray(glob.glob(os.path.join(se_dir,"*.*")))

    psf_files = []
    psf_dir = os.path.expandvars('$GFS_PYTHON_CODE/sunpy_dev/kernels')
    #psf_names = ['PSFSTD_ACSWFC_F435W.fits','PSFSTD_ACSWFC_F606W.fits','PSFSTD_ACSWFC_F775W_SM3.fits','PSFSTD_ACSWFC_F850L_SM3.fits',
    #             'PSFSTD_WFC3IR_F105W.fits','PSFSTD_WFC3IR_F125W.fits','PSFSTD_WFC3IR_F160W.fits',
    #             'PSF_NIRCam_F070W_revV-1.fits','PSF_NIRCam_F090W_revV-1.fits','PSF_NIRCam_F115W_revV-1.fits','PSF_NIRCam_F150W_revV-1.fits',
    #             'PSF_NIRCam_F200W_revV-1.fits','PSF_NIRCam_F277W_revV-1.fits','PSF_NIRCam_F356W_revV-1.fits','PSF_NIRCam_F444W_revV-1.fits',
    #             'PSFSTD_WFC3IR_F140W.fits','PSFSTD_WFC3UV_F275W.fits','PSFSTD_WFC3UV_F336W.fits','PSFSTD_ACSWFC_F814W.fits']

    psf_names = ['TinyTim_IllustrisPSFs/F435W_rebin.fits','TinyTim_IllustrisPSFs/F606W_rebin.fits','TinyTim_IllustrisPSFs/F775W_rebin.fits','TinyTim_IllustrisPSFs/F850LP_rebin.fits',
                 'TinyTim_IllustrisPSFs/F105W_rebin.fits','TinyTim_IllustrisPSFs/F125W_rebin.fits','TinyTim_IllustrisPSFs/F160W_rebin.fits',
                 'WebbPSF_F070W_trunc.fits','WebbPSF_F090W_trunc.fits','WebbPSF_F115W_trunc.fits','WebbPSF_F150W_trunc.fits',
                 'WebbPSF_F200W_trunc.fits','WebbPSF_F277W_trunc.fits','WebbPSF_F356W_trunc.fits','WebbPSF_F444W_trunc.fits',
                 'TinyTim_IllustrisPSFs/F140W_rebin.fits','TinyTim_IllustrisPSFs/F275W_rebin.fits','TinyTim_IllustrisPSFs/F336W_rebin.fits','TinyTim_IllustrisPSFs/F814W_rebin.fits']

    #psf_pix_arcsec = [0.0125,0.0125,0.0125,0.0125,0.0325,0.0325,0.0325,0.007925,0.007925,0.007925,0.007925,0.007925,0.0162,0.0162,0.0162,0.0325,0.0100,0.0100,0.0125]
    #switch to JWST detector sampling for efficiency.  They're model psfs anyway, full accuracy not essential

    psf_pix_arcsec = [0.03,0.03,0.03,0.03,0.06,0.06,0.06,0.0317,0.0317,0.0317,0.0317,0.0317,0.0648,0.0648,0.0648,0.06,0.03,0.03,0.03]
    psf_truncate = [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]
    psf_hdu_num = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    psf_fwhm = [0.10,0.11,0.12,0.13,0.14,0.17,0.20,0.11,0.11,0.11,0.11,0.12,0.15,0.18,0.25,0.18,0.07,0.08,0.13]
    #these settings yield full subhalo (4 cams) convolution in 0.92s!  convolve_fft ftw!

    for pname in psf_names:
        psf_file = os.path.join(psf_dir,pname)
        psf_files.append(psf_file)
        print psf_file, os.path.lexists(psf_file)

    ###  PSFSTD; WFC3 = 0.06 arcsec, ACS = 0.03 arcsec... I think
    ### NIRCAM in header with keyword 'PIXELSCL';  short 0.07925 long 0.0162
    ## acs wfc 0.05 arcsec pixels... PSFSTD x4 oversample?
    ## wfc3 ir 0.13 arcsec
    ## wfc3 uv 0.04 arcsec

    mockimage_parameters = analysis_parameters('mockimage_default')
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

    #use exactly one detection and segmentation per object, depending on redshift
    #enormous simplification
    #observationally, go w deepest filter.  here... ?

    mockimage_parameters.segment_filter_label = seg_filter_label
    mockimage_parameters.segment_filter_index = np.where(np.asarray(mockimage_parameters.filter_labels) == seg_filter_label)[0][0]

    print mockimage_parameters.segment_filter_label
    print mockimage_parameters.segment_filter_index
    
    assert(len(psf_pix_arcsec)==len(pixsize_arcsec))
    assert(len(filter_labels)==len(mockimage_parameters.psf_files))

    
    for i,bbfile in enumerate(bbfile_list):

        try:
            process_single_broadband(bbfile,mockimage_parameters,clobber=clobber,do_idl=do_idl,analyze=analyze)
        except (KeyboardInterrupt,NameError,AttributeError,KeyError,TypeError) as e:
            print e
            raise
        except:
            print "Exception while processing broadband: ", bbfile
            print "Error:", sys.exc_info()[0]
        else:
            print "Successfully processed broadband: ", bbfile

    os.chdir(cwd)

    return 1


def process_snapshot(snap_path='.',clobber=False,max=None,maxper=None,starti=0,seg_filter_label='NC-F200W',do_idl=False,analyze=True,**kwargs):

    cwd = os.path.abspath(os.curdir)

    os.chdir(snap_path)

    subdir_list = np.sort(np.asarray(glob.glob('subdir_???')))
    print subdir_list

    for i,sub in enumerate(subdir_list[starti:]):
        if max != None:
            if i >= max:
                continue

        try:
            result = process_subdir(sub,clobber=clobber,max=maxper,seg_filter_label=seg_filter_label,do_idl=do_idl,analyze=analyze,**kwargs)
        except:
            print "Exception while processing subdir: ", sub
            print "Error:", sys.exc_info()[0]
        else:
            print "Successfully processed subdir: ", sub
    

    os.chdir(cwd)
    return 1


def test_10107():
    result = process_subdir('/Users/gsnyder/Dropbox/Projects/snapshot_060/subdir_000',clobber=True,analyze=False,galaxy='broadband_red_10107.fits.gz',seg_filter_label='NC-F200W',magsb_limits=[25],camindices=[0,1,2,3],do_idl=False)
    return

def run_060_000():
    #result = process_subdir('/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/snapshot_060/subdir_000',clobber=True,galaxy='broadband_red_0.fits.gz',seg_filter_label='NC-F200W',magsb_limits=[25])
    result = process_subdir('/Users/gsnyder/Dropbox/Projects/snapshot_060/subdir_000',clobber=False,seg_filter_label='NC-F200W',magsb_limits=[25],do_idl=False)
    #result = process_subdir('/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/snapshot_060/subdir_000',clobber=True,seg_filter_label='NC-F200W',magsb_limits=[25],do_idl=True)

    return

###
### Finished creating images 035-054
def run_035(starti=0,max=None):
    result = process_snapshot('/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/snapshot_035',clobber=False,analyze=True,seg_filter_label='NC-F444W',magsb_limits=[25,27,29],do_idl=False,starti=starti,max=max)
    return
def run_038(starti=0,max=None):
    result = process_snapshot('/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/snapshot_038',clobber=False,analyze=True,seg_filter_label='NC-F444W',magsb_limits=[25,27,29],do_idl=False,starti=starti,max=max)
    return
def run_041(starti=0,max=None):
    result = process_snapshot('/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/snapshot_041',clobber=False,analyze=True,seg_filter_label='NC-F444W',magsb_limits=[25,27,29],do_idl=False,starti=starti,max=max)
    return
def run_045(starti=0,max=None):
    result = process_snapshot('/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/snapshot_045',clobber=False,analyze=True,seg_filter_label='NC-F444W',magsb_limits=[25,27,29],do_idl=False,starti=starti,max=max)
    return
def run_049(starti=0,max=None):
    result = process_snapshot('/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/snapshot_049',clobber=False,analyze=True,seg_filter_label='NC-F444W',magsb_limits=[25,27,29],do_idl=False,starti=starti,max=max)
    return
def run_054(starti=0,max=None):
    #result = process_snapshot('/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/snapshot_054',         clobber=False,analyze=True,seg_filter_label='NC-F444W',magsb_limits=[25,27],do_idl=False,starti=starti,max=max)
    result = process_subdir('/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/snapshot_054/subdir_000',clobber=False,analyze=True,seg_filter_label='NC-F444W',magsb_limits=[29],   do_idl=False)
    return
def run_060(starti=0,max=None):
    #result = process_snapshot('/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/snapshot_060',         clobber=False,analyze=True,seg_filter_label='NC-F200W',magsb_limits=[25,27],do_idl=False,starti=starti,max=max)
    result = process_subdir('/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/snapshot_060/subdir_000',clobber=False,analyze=True,seg_filter_label='NC-F200W',magsb_limits=[29],   do_idl=False)
    return

def run_064(starti=0,max=None):
    #result = process_snapshot('/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/snapshot_064',         clobber=False,analyze=True,seg_filter_label='NC-F200W',magsb_limits=[25,27],do_idl=False,starti=starti,max=max)
    result = process_subdir('/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/snapshot_064/subdir_000',clobber=False,analyze=True,seg_filter_label='NC-F200W',magsb_limits=[29],   do_idl=False)
    return

def run_068(starti=0,max=None):
    #result = process_snapshot('/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/snapshot_068',         clobber=False,analyze=True,seg_filter_label='NC-F200W',magsb_limits=[25],   do_idl=False,starti=starti,max=max)
    result = process_subdir('/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/snapshot_068/subdir_000',clobber=False,analyze=True,seg_filter_label='NC-F200W',magsb_limits=[27,29],do_idl=False)
    return


def run_075(starti=0,max=None):
    #result = process_snapshot('/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/snapshot_075',         clobber=False,analyze=True,seg_filter_label='NC-F200W',magsb_limits=[25],   do_idl=False,starti=starti,max=max)
    result = process_subdir('/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/snapshot_075/subdir_000',clobber=False,analyze=True,seg_filter_label='NC-F200W',magsb_limits=[27,29],do_idl=False)
    return
def run_085(starti=0,max=None):
    #result = process_snapshot('/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/snapshot_085',         clobber=False,analyze=True,seg_filter_label='NC-F200W',magsb_limits=[25],   do_idl=False,starti=starti,max=max)
    result = process_subdir('/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/snapshot_085/subdir_000',clobber=False,analyze=True,seg_filter_label='NC-F200W',magsb_limits=[27,29],do_idl=False)
    return

###
###


def run_103(starti=0,max=None):
    #result = process_snapshot('/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/snapshot_103',         clobber=False,analyze=True,seg_filter_label='ACS-F850LP',magsb_limits=[25],   do_idl=False,starti=starti,max=max)
    result = process_subdir('/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/snapshot_103/subdir_000',clobber=False,analyze=True,seg_filter_label='ACS-F850LP',magsb_limits=[27,29],do_idl=False)
    return

if __name__=="__main__":

    print sys.argv

    if len(sys.argv) == 2:
        method = sys.argv[1]
        statsf = 'profiler_stats_'+method
        print "Running "+method+'()'
        print "Profiling "+statsf
        cProfile.run(method+'()',statsf)
        p = pstats.Stats(statsf)
        p.strip_dirs().sort_stats('time').print_stats(45)
    elif len(sys.argv) == 4:
        method = sys.argv[1]
        start = sys.argv[2]
        maxa = sys.argv[3]
        runstr = method+'(starti='+start+',max='+maxa+')'
        print runstr
        exec runstr
    else:
        print "Usage:  setup_illustris_morphs.py 'methodstring' OR "
        print "        setup_illustris_morphs.py 'methodstring' starti max"


    #cProfile.run('run_060_000()','profiler_stats_060_000')
    #p = pstats.Stats('profiler_stats_060_000')
    #p.strip_dirs().sort_stats('time').print_stats(45)


    #cProfile.run('test_10107()','profiler_stats_10107')
    #p = pstats.Stats('profiler_stats_10107')
    #p.strip_dirs().sort_stats('time').print_stats(45)

    #redshift segmentation filter selection:
    ##  z=0.5,0           F850LP ?
    ##  z=1,1.5,2,2.5,3   F200W
    ##  z=4,5,6,7,8,9     F444W
