import cProfile
import pstats
import math
import string
import sys
import struct
import matplotlib
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
import statmorph
import datetime
import setup_synthetic_images_mp as ssimp

# Based on candelize.py

def process_single_broadband(bbfile, analysis_object, bbase='broadband_red_',
            clobber=False, analyze=True, do_idl=False, Np=2, maxq=10000,
            lim=None, zip_after=True):
    #create subdirectory to hold mock images and analyses
    if bbase is not "broadbandz":
        sh_id = bbfile[len(bbase):].rstrip('.fits')
        bb_dir = 'images_subhalo_'+sh_id
        subdir_path = os.path.dirname(os.path.abspath(bbfile))
        print('subdir_path is %s' % (subdir_path))
        subdirnum = subdir_path[-3:]
        snap_path = os.path.dirname(subdir_path)
        snapnum = snap_path[-3:]
        snap_prefix = 'snap'+snapnum+'dir'+subdirnum+'sh'+sh_id
    else:
        snapnum=None
        subdirnum=None
        sh_id=None
        if analysis_object.use_nonscatter is True:
            snap_prefix = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(bbfile))))+'_nonscatter_'
        else:
            snap_prefix = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(bbfile))))+'_'
        bb_dir = 'images_'+snap_prefix.rstrip('_')

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
            except (KeyboardInterrupt,NameError,AttributeError,TypeError,IndexError,KeyError) as e:
                print e
                raise
            except:
                print "Exception while processing filter image creation: ", filter_label, custom_filename_sb00
                print "Error:", sys.exc_info()[0]


    if openlist is not None:
        openlist.close()
    

    #compress when finished with broadband.fits file
    if zip_after is True:
        if is_unzipped is True:
            subprocess.call(['pigz', '-9', '-p', str(Np), bbfile])


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


            finished_objs = process_mag(analysis_object,bb_dir,snap_prefix,camstring,maglim,analyze,
                                        segmap_filename,seg_image,seg_header,seg_npix,seg_filter_label,
                                        clabel,cpos0,cpos1,figure,nx,ny,clobber,idl_input_file,py_output_file,
                                        Np=Np,maxq=maxq,lim=lim)


            for i,seq in enumerate(finished_objs):
                print seq
                res = plot_test_stamp(seq[0],figure,nx,ny,seq[1])
                

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


    if bbase is 'broadbandz':
        subprocess.call(['tar','cf',bb_dir+'.tar',bb_dir])
        subprocess.call(['rm','-rf',bb_dir])
        
    return bb_dir




def process_snapshot(subdirpath='.', clobber=False, galaxy=None,
        seg_filter_label='panstarrs_ps1_g', magsb_limits=[20.0,22.0],
        camindices=[0,1,2,3], do_idl=False, analyze=False, use_nonscatter=True, Np=2):

    cwd = os.path.abspath(os.curdir)

    os.chdir(subdirpath)

    bbfile_list = np.sort(np.asarray(glob.glob('broadbandz.fits*')))   #enable reading .fits.gz files
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

    filters_to_analyze = [
            'panstarrs/panstarrs_ps1_g',
            'panstarrs/panstarrs_ps1_open',
            'lsst/lsst_u',
            'lsst/lsst_y3']

    skip_filter_boolean = [
            False,
            False,
            False,
            False]

    print(filters_to_analyze)
    
    # A bit of oversampling:
    pixsize_arcsec = [
            0.2,
            0.2,
            0.2,
            0.2]
    
    filter_labels = [
            'panstarrs_ps1_g',
            'panstarrs_ps1_open',
            'lsst_u',
            'lsst_y3']

    filter_indices = []

    print(len(filters_to_analyze), len(skip_filter_boolean), len(filter_labels))
    
    for i,f in enumerate(filters_to_analyze):
        fi = np.where(fils==f)
        print(fi[0][0], f, fils[fi[0][0]], filter_labels[i]) #, filters_to_analyze[fi]
        filter_indices.append(fi[0][0])

    filter_indices = np.asarray(filter_indices)

    print(filter_indices)

    # order of filter_labels in wavelength space
    filter_lambda_order = [2, 0, 1, 3]

    # References:
    # http://svo2.cab.inta-csic.es/svo/theory/fps/index.php?mode=browse&gname=PAN-STARRS
    # http://svo2.cab.inta-csic.es/svo/theory/fps/index.php?mode=browse&gname=LSST

    #photfnu units Jy; flux in 1 ct/s (definitely incorrect values)
    photfnu_Jy = [1e-7, 1e-7, 1e-7, 1e-7]
    
    #morphcode_dir = "/Users/gsnyder/Documents/pro/morph_december2013/morph_pro/"
    #morphcode_files = np.asarray(glob.glob(os.path.join(morphcode_dir,"*.*")))

    #se_dir = '/Users/gsnyder/Documents/Projects/Illustris_Morphology/Illustris-CANDELS/SE_scripts'
    #se_files = np.asarray(glob.glob(os.path.join(se_dir,"*.*")))


    psf_dir = '/home/vrg/filter_data/psf'
    psf_names = [
            'gauss_fwhm_5_pixels.fits',
            'gauss_fwhm_5_pixels.fits',
            'gauss_fwhm_5_pixels.fits',
            'gauss_fwhm_5_pixels.fits']

    # A bit of oversampling:
    psf_pix_arcsec = [0.2, 0.2, 0.2, 0.2]
    psf_truncate = [None,None,None,None]
    psf_hdu_num = [0, 0, 0, 0]
    psf_fwhm = [1.0, 1.0, 1.0, 1.0]

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
    #mockimage_parameters.morphcode_base = morphcode_dir
    #mockimage_parameters.morphcode_files = morphcode_files
    #mockimage_parameters.se_base = se_dir
    #mockimage_parameters.se_files = se_files
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

    mockimage_parameters.segment_filter_label = seg_filter_label
    mockimage_parameters.segment_filter_index = np.where(np.asarray(mockimage_parameters.filter_labels) == seg_filter_label)[0][0]

    print(mockimage_parameters.segment_filter_label)
    print(mockimage_parameters.segment_filter_index)
    
    assert(len(psf_pix_arcsec)==len(pixsize_arcsec))
    assert(len(filter_labels)==len(mockimage_parameters.psf_files))

    bbdirs = []
    
    for i,bbfile in enumerate(bbfile_list):

        try:
            bbdir = process_single_broadband(bbfile,mockimage_parameters,clobber=clobber,do_idl=do_idl,analyze=analyze,bbase="broadbandz",Np=Np)
            bbdirs.append(bbdir)
        except (KeyboardInterrupt,NameError,AttributeError,KeyError,TypeError,IndexError) as e:
            print(e)
            raise
        except:
            print("Exception while processing broadband: ", bbfile)
            print("Error:", sys.exc_info()[0])
        else:
            print("Successfully processed broadband: ", bbfile)

    os.chdir(cwd)

    return bbdirs



if __name__=="__main__":
    
    # Without dust
    res = process_snapshot(subdirpath='.', seg_filter_label='panstarrs_ps1_g',
            magsb_limits=[20.0,22.0], camindices=[0,1,2,3],
            do_idl=False, analyze=False, use_nonscatter=True, Np=4)
    #~ # Include dust
    #~ res = process_snapshot(subdirpath='.', seg_filter_label='panstarrs_ps1_g',
            #~ magsb_limits=[20.0,22.0], camindices=[0,1,2,3],
            #~ do_idl=False, analyze=True, use_nonscatter=False, Np=4)

    
