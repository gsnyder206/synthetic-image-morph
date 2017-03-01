import cProfile
import pstats
import math
import string
import sys
import struct
import matplotlib
#matplotlib.use('PDF')
import matplotlib.pyplot as pyplot
import matplotlib.colors as pycolors
import matplotlib.cm as cm
#import matplotlib.patches as patches
import numpy as np
#import cPickle
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
#import cosmocalc
#import congrid
import astropy.io.ascii as ascii
import warnings
import subprocess
import photutils
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel
from astropy.visualization.mpl_normalize import ImageNormalize
import astropy.cosmology
#from astropy.visualization import *
import astropy.io.fits as pyfits
#import statmorph
import copy
import medianstats_bootstrap as msbs
import illustris_python as ilpy
import h5py
import pickle as cPickle



class morphdata:
    def __init__(self,snapshot_directory,camlist,filters,depths):
        self.snapdir = snapshot_directory
        self.camlist = camlist
        self.filters = filters
        self.depths = depths
        self.snapstr = os.path.basename(self.snapdir)[-3:]
        self.morph_headers = None
        self.si_headers = None
        self.seg_headers = None
        self.pum_headers = None



    #given an extension name and header card, return np array values for each subhalo, cam, filter, and depth
    def parse_values(self,extname='LotzMorphMeasurements',card='GINI',missing=-9.0):

        ncams = (self.camlist).shape[0]
        nfilters = (self.filters).shape[0]
        ndepths = (self.depths).shape[0]
        nsubhalos = (self.unique_subhalo_ids).shape[0]
        
        #non-critical value here
        #nsubdirs = len(self.subdir_list) 

        assert (extname=='LotzMorphMeasurements') or (extname=='SYNTHETIC_IMAGE') or (extname=='PhotUtilsMeasurements')

        value_array = np.zeros((nsubhalos,ncams,nfilters,ndepths),dtype=np.float32) + missing

        value_dict = {}

        for cs in card:
            value_dict[cs] = np.zeros((nsubhalos,ncams,nfilters,ndepths),dtype=np.float32) + missing


        if (self.morph_headers is None) and extname == 'LotzMorphMeasurements':
            self.morph_headers = []
            #loop over subdirs to find morph measurements
            value_dict = self.parse_loop_append(extname,card,self.morph_headers,value_array,value_dict)
        elif (self.si_headers is None) and extname == 'SYNTHETIC_IMAGE':
            self.si_headers = []
            #loop over subdirs to find morph measurements
            value_dict = self.parse_loop_append(extname,card,self.si_headers,value_array,value_dict)
        elif (self.pum_headers is None) and extname == 'PhotUtilsMeasurements':
            self.pum_headers = []
            #loop over subdirs to find morph measurements
            value_dict = self.parse_loop_append(extname,card,self.pum_headers,value_array,value_dict)

        elif extname == 'LotzMorphMeasurements':
            #loop over subdirs to find morph measurements
            value_dict = self.parse_loop_fast(extname,card,self.morph_headers,value_array,value_dict)
        elif extname == 'SYNTHETIC_IMAGE':
            #loop over subdirs to find morph measurements
            value_dict = self.parse_loop_fast(extname,card,self.si_headers,value_array,value_dict)
        elif extname == 'PhotUtilsMeasurements':
            value_dict = self.parse_loop_fast(extname,card,self.pum_headers,value_array,value_dict)
            
        return value_dict



    def parse_loop_fast(self,extname,card,header_list,value_array,value_dict):
        for i,imfile in enumerate(self.image_files):
            this_dict = self.value_dicts[i]
            if this_dict is None:
                pass
            else:
                for cs in card:
                    try:
                        this_value = this_dict[cs]
                    except KeyError:
                        pass
                    else:
                        this_i = self.unique_subhalo_inverse[i]
                        this_j = self.cam_indices[i]
                        this_k = self.fil_indices[i]
                        this_l = 0
                        value_dict[cs][this_i,this_j,this_k,this_l] = this_value
                        #value_array[this_i,this_j,this_k,this_l] = this_value

        return value_dict

    def parse_loop_use(self,extname,card,header_list,value_array,value_dict):
        for i,imfile in enumerate(self.image_files):
            header = header_list[i]
            if header is None:
                pass
            else:
                this_header = header
                for cs in card:
                    try:
                        this_value = this_header[cs]
                    except KeyError:
                        pass
                    else:
                        this_i = self.unique_subhalo_inverse[i]
                        this_j = self.cam_indices[i]
                        this_k = self.fil_indices[i]
                        this_l = 0
                        value_dict[cs][this_i,this_j,this_k,this_l] = this_value
                        #value_array[this_i,this_j,this_k,this_l] = this_value

        return value_dict

    def parse_loop_append(self,extname,card,header_list,value_array,value_dict):
        for i,imfile in enumerate(self.image_files):
            hdulist = pyfits.open(imfile)
            try:
                this_hdu = hdulist[extname]
            except KeyError:
                this_header = None
                header_list.append(this_header)
            else:
                this_header = this_hdu.header
                header_list.append(this_header)
                for cs in card:
                    try:
                        this_value = this_header[cs]
                    except KeyError:
                        pass
                    else:
                        this_i = self.unique_subhalo_inverse[i]
                        this_j = self.cam_indices[i]
                        this_k = self.fil_indices[i]
                        this_l = 0
                        value_dict[cs][this_i,this_j,this_k,this_l] = this_value
                        #value_array[this_i,this_j,this_k,this_l] = this_value
        return value_dict

    def parse_idl_values(self,col='col22',missing=-9.0):

        ncams = (self.camlist).shape[0]
        nfilters = (self.filters).shape[0]
        ndepths = (self.depths).shape[0]
        nsubhalos = (self.unique_subhalo_ids).shape[0]
        
        #non-critical value here
        #nsubdirs = len(self.subdir_list) 

        value_array = np.zeros((nsubhalos,ncams,nfilters,ndepths),dtype=np.float32) + missing

        last_i = -1

        for i,imfile in enumerate(self.image_files):
            this_i = self.unique_subhalo_inverse[i]
            this_j = self.cam_indices[i]
            this_k = self.fil_indices[i]
            this_l = 0

            subdir = self.subdirs[i]
            imdir = 'images_subhalo_'+self.subhalo_ids[i]
            idlfile = os.path.join(os.path.join(subdir,imdir),self.idlfiles[i])
            #print idlfile
            if not os.path.lexists(idlfile):
                continue

            if this_i != last_i:
                try:
                    data = ascii.read(idlfile,data_start=1)
                except InconsistentTableError:
                    pass
                else:
                    last_i = this_i
            else:
                data = data

            column = data[col]
            filenames = data['col1']
            filematch = os.path.join(imdir,os.path.basename(imfile))
            ind = np.where(filenames==filematch)[0]
            #print ind
            if ind.shape[0] != 1:
                continue

            this_value = column[ind[0]]
            value_array[this_i,this_j,this_k,this_l]=this_value

        return value_array

    def parse_image_files(self, array, sh_array=None, cards=None):
        self.old_image_files = array
        self.snapnums = []
        self.subdirs = []
        self.subhalo_ids = []
        self.cameras = []
        self.depth_values = []
        self.flabels = []
        self.new_image_files = []
        self.cam_indices = []
        self.fil_indices = []
        self.idlfiles = []

        self.si_headers = []
        self.pum_headers = []
        self.morph_headers = []
        self.value_dicts = []

        #loop over subdirs to find morph measurements
        for i,imfile in enumerate(self.old_image_files):
            hdulist = pyfits.open(imfile)
            header = hdulist[0].header
            camstr = header['CAMERA']
            filstr = imfile.split('_')[-2]
            substr = header['SUBH_ID']
            cam_index = np.where(self.camlist==np.int32(camstr))[0]
            fil_index = np.where(self.filters==filstr)[0]
            if sh_array is not None:
                sh_index = np.where(sh_array==np.int32(substr))[0]
                if cam_index.shape[0]==0 or fil_index.shape[0]==0 or sh_index.shape[0]==0:
                    continue
            else:
                if cam_index.shape[0]==0 or fil_index.shape[0]==0:
                    continue                

            self.new_image_files.append(imfile)
            self.snapnums.append(header['SNAPNUM'])
            self.subdirs.append('subdir_'+header['SUBDIR'])
            self.subhalo_ids.append(substr)
            self.cameras.append(header['CAMERA'])
            self.depth_values.append(imfile[-9:-5])
            #self.flabels.append(header['FLABEL'])
            self.flabels.append(imfile.split('_')[-2])
            self.fil_indices.append(fil_index[0])
            self.cam_indices.append(cam_index[0])
            #self.si_headers.append(header)

            value_dict = {}
            if cards is not None:

                try:
                    pum_header = hdulist['PhotUtilsMeasurements'].header
                except KeyError:
                    pass
                else:
                    for c in pum_header.cards:
                        header.append(c)

                try:
                    morph_header = hdulist['LotzMorphMeasurements'].header
                except KeyError:
                    pass
                else:
                    for c in morph_header.cards:
                        header.append(c)    
                    
                for cs in cards:
                    try:
                        value_dict[cs] = header[cs]
                    except KeyError:
                        pass
                    
            else:
                value_dict = None


            self.value_dicts.append(value_dict)

            self.idlfiles.append('snap'+self.snapnums[-1]+'dir'+header['SUBDIR']+'sh'+self.subhalo_ids[-1]+'_idloutput.txt')

            hdulist.close()
            #if i % 100 ==0:
            #    print imfile, self.snapnums[-1], self.subdirs[-1], self.subhalo_ids[-1], self.cameras[-1], self.depths[-1], self.flabels[-1]


        self.image_files = np.asarray(self.new_image_files)
        self.snapnums = np.asarray(self.snapnums)
        self.subdirs = np.asarray(self.subdirs)
        self.subhalo_ids = np.asarray(self.subhalo_ids)
        self.cameras = np.asarray(self.cameras)
        self.depth_values = np.asarray(self.depth_values)
        self.flabels = np.asarray(self.flabels)
        self.cam_indices = np.asarray(self.cam_indices)
        self.fil_indices = np.asarray(self.fil_indices)
        self.idlfiles = np.asarray(self.idlfiles)

        self.unique_subhalo_ids,self.unique_subhalo_indices,self.unique_subhalo_inverse = np.unique(self.subhalo_ids,return_index=True,return_inverse=True)

        return




#unpack critical values, store in an object, save to disk and/or return to user
def parse_illustris_morph_snapshot(snapshot_directory='/Users/gsnyder/Dropbox/Projects/snapshot_060',sh_array=None,get_idl=False,depth='SB25',dofull=False):
    if not os.path.lexists(snapshot_directory):
        print("Error parsing Illustris morphs: directory doesn't exist: "+snapshot_directory)

    cwd = os.path.abspath(os.curdir)
    os.chdir(snapshot_directory)

    camlist = np.asarray([0,1,2,3])
    filters = np.asarray(['WFC3-F275W','WFC3-F336W','ACS-F435W','ACS-F606W','ACS-F775W','ACS-F814W','ACS-F850LP',
                          'WFC3-F105W','WFC3-F125W','WFC3-F140W','WFC3-F160W',
                          'NC-F070W','NC-F090W','NC-F115W','NC-F150W','NC-F200W','NC-F277W','NC-F356W','NC-F444W'])
    #filters = np.asarray(['NC-F200W','NC-F277W'])

    #print filters

    depths = np.asarray([depth])
    
    assert depths.shape[0]==1


    si_cards = np.asarray(['PIXSCALE','SUABSMAG','RMS','APROXPSF','REDSHIFT','EFLAMBDA'])
    pum_cards = np.asarray(['SEGMAG','SEGMAGE','AREA','ECCENT','ELLIPT','EQ_RAD','SMAJSIG','SMINSIG','XMIN','XMAX','YMIN','YMAX'])
    morph_cards = np.asarray(['GINI','M20','CC','CC_ERR','RPE','RPE_ERR','R5C','ER5C','R5E','ER5E',
                   'SNP','ASYM','CC_R20','CC_R80','ELONG','ORIENT','M_A','M_I2','RPC','FS93_I1','FS93_I2',
                   'FS93_I3','FS93_I4','Hu_I1','Hu_I2','Hu_I3','Hu_I4','Hu_I5','Hu_I6','Hu_I7','Hu_I8','FLAG','CFLAG','AXC','AYC','MXC','MYC',
                   'MID_MP','MID_A1','MID_A2','MID_LEV','MID_I','MID_IXP','MID_IYP','MID_D','MID_DA','MID_GINI','MID_M20','MID_AREA','MID_SNP',
                   'MID2_MP','MID2_A1','MID2_A2','MID2_LEV','MID2_I','MID2_IXP','MID2_IYP','MID2_D','MID2_DA'])

    all_cards = np.append(np.append(si_cards,pum_cards),morph_cards)


    morph_data_obj = morphdata(snapshot_directory,camlist,filters,depths)
    morph_data_obj.parse_image_files(np.sort(np.asarray(glob.glob('subdir_???/images_subhalo_*/snap???dir???sh*cam??_*_'+depths[0]+'.fits'))),sh_array=sh_array,cards=all_cards)

    #print morph_data_obj.image_files.shape
    #print morph_data_obj.cam_indices.shape
    #print morph_data_obj.unique_subhalo_inverse.shape
    #print morph_data_obj.fil_indices.shape
    #print morph_data_obj.old_image_files.shape

    return_dict = morph_data_obj.parse_values(extname='SYNTHETIC_IMAGE',card=si_cards)

    morph_data_obj.si_dict = return_dict
    morph_data_obj.pix_arcsec =return_dict['PIXSCALE']
    morph_data_obj.sunrise_absmag = return_dict['SUABSMAG']
    morph_data_obj.rms = return_dict['RMS']
    morph_data_obj.approx_psf_fwhm_arcsec = return_dict['APROXPSF']
    morph_data_obj.redshift = return_dict['REDSHIFT']
    morph_data_obj.eflambda_um = return_dict['EFLAMBDA']


    #print "Parsed SI cards"

    return_dict = morph_data_obj.parse_values(extname='PhotUtilsMeasurements',card=pum_cards)
    morph_data_obj.pum_dict = return_dict

    morph_data_obj.magseg = return_dict[pum_cards[0]] #morph_data_obj.parse_values(extname='PhotUtilsMeasurements',card='SEGMAG')
    morph_data_obj.magseg_err = return_dict[pum_cards[1]] #morph_data_obj.parse_values(extname='PhotUtilsMeasurements',card='SEGMAGE')
    morph_data_obj.seg_area = return_dict[pum_cards[2]] #morph_data_obj.parse_values(extname='PhotUtilsMeasurements',card='AREA')
    morph_data_obj.seg_eccent = return_dict[pum_cards[3]] #morph_data_obj.parse_values(extname='PhotUtilsMeasurements',card='ECCENT')
    morph_data_obj.seg_ellipt = return_dict[pum_cards[4]] #morph_data_obj.parse_values(extname='PhotUtilsMeasurements',card='ELLIPT')
    morph_data_obj.seg_eqrad = return_dict[pum_cards[5]] #morph_data_obj.parse_values(extname='PhotUtilsMeasurements',card='EQ_RAD')
    morph_data_obj.seg_smajsig = return_dict[pum_cards[6]] #morph_data_obj.parse_values(extname='PhotUtilsMeasurements',card='SMAJSIG')
    morph_data_obj.seg_sminsig = return_dict[pum_cards[7]] #morph_data_obj.parse_values(extname='PhotUtilsMeasurements',card='SMINSIG')
    morph_data_obj.seg_xmin = return_dict[pum_cards[8]] #morph_data_obj.parse_values(extname='PhotUtilsMeasurements',card='XMIN')
    morph_data_obj.seg_xmax = return_dict[pum_cards[9]] #morph_data_obj.parse_values(extname='PhotUtilsMeasurements',card='XMAX')
    morph_data_obj.seg_ymin = return_dict[pum_cards[10]] #morph_data_obj.parse_values(extname='PhotUtilsMeasurements',card='YMIN')
    morph_data_obj.seg_ymax = return_dict[pum_cards[11]] #morph_data_obj.parse_values(extname='PhotUtilsMeasurements',card='YMAX')

    #print "Parsed PUM cards"

    return_dict = morph_data_obj.parse_values(card=morph_cards)
    morph_data_obj.morph_dict = return_dict

    morph_data_obj.gini = return_dict[morph_cards[0]] #morph_data_obj.parse_values(card='GINI')
    morph_data_obj.m20 = return_dict[morph_cards[1]] #morph_data_obj.parse_values(card='M20')
    morph_data_obj.cc = return_dict[morph_cards[2]] #morph_data_obj.parse_values(card='CC')
    morph_data_obj.cc_err = return_dict[morph_cards[3]] #morph_data_obj.parse_values(card='CC_ERR')
    morph_data_obj.rpe = return_dict[morph_cards[4]] #morph_data_obj.parse_values(card='RPE')
    morph_data_obj.rpe_err = return_dict[morph_cards[5]] #morph_data_obj.parse_values(card='RPE_ERR')
    morph_data_obj.rhalfc = return_dict[morph_cards[6]] #morph_data_obj.parse_values(card='R5C')
    morph_data_obj.rhalfc_err = return_dict[morph_cards[7]] #morph_data_obj.parse_values(card='R5E')
    morph_data_obj.rhalfe = return_dict[morph_cards[8]] #morph_data_obj.parse_values(card='ER5C')
    morph_data_obj.rhalfe_err = return_dict[morph_cards[9]] #morph_data_obj.parse_values(card='ER5E')

    morph_data_obj.snpix = return_dict[morph_cards[10]] #morph_data_obj.parse_values(card='SNP')
    morph_data_obj.asym = return_dict[morph_cards[11]] #morph_data_obj.parse_values(card='ASYM')
    morph_data_obj.r20 = return_dict[morph_cards[12]] #morph_data_obj.parse_values(card='CC_R20')
    morph_data_obj.r80 = return_dict[morph_cards[13]] #morph_data_obj.parse_values(card='CC_R80')
    morph_data_obj.elong = return_dict[morph_cards[14]] #morph_data_obj.parse_values(card='ELONG')
    morph_data_obj.orient = return_dict[morph_cards[15]] #morph_data_obj.parse_values(card='ORIENT')
    morph_data_obj.m_a = return_dict[morph_cards[16]] #morph_data_obj.parse_values(card='M_A')
    morph_data_obj.m_i2 = return_dict[morph_cards[17]] #morph_data_obj.parse_values(card='M_I2')
    morph_data_obj.rpc = return_dict[morph_cards[18]] #morph_data_obj.parse_values(card='RPC')
    morph_data_obj.fs93_i1 = return_dict[morph_cards[19]] #morph_data_obj.parse_values(card='FS93_I1')
    morph_data_obj.fs93_i2 = return_dict[morph_cards[20]] #morph_data_obj.parse_values(card='FS93_I2')
    morph_data_obj.fs93_i3 = return_dict[morph_cards[21]] #morph_data_obj.parse_values(card='FS93_I3')
    morph_data_obj.fs93_i4 = return_dict[morph_cards[22]] #morph_data_obj.parse_values(card='FS93_I4')
    morph_data_obj.hu_i1 = return_dict[morph_cards[23]] #morph_data_obj.parse_values(card='Hu_I1')
    morph_data_obj.hu_i2 = return_dict[morph_cards[24]] #morph_data_obj.parse_values(card='Hu_I2')
    morph_data_obj.hu_i3 = return_dict[morph_cards[25]] #morph_data_obj.parse_values(card='Hu_I3')
    morph_data_obj.hu_i4 = return_dict[morph_cards[26]] #morph_data_obj.parse_values(card='Hu_I4')
    morph_data_obj.hu_i5 = return_dict[morph_cards[27]] #morph_data_obj.parse_values(card='Hu_I5')
    morph_data_obj.hu_i6 = return_dict[morph_cards[28]] #morph_data_obj.parse_values(card='Hu_I6')
    morph_data_obj.hu_i7 = return_dict[morph_cards[29]] #morph_data_obj.parse_values(card='Hu_I7')
    morph_data_obj.hu_i8 = return_dict[morph_cards[30]] #morph_data_obj.parse_values(card='Hu_I8')
    morph_data_obj.flag = return_dict[morph_cards[31]] #morph_data_obj.parse_values(card='FLAG')
    morph_data_obj.cflag = return_dict[morph_cards[32]] #morph_data_obj.parse_values(card='CFLAG')
    morph_data_obj.axc = return_dict[morph_cards[33]] #morph_data_obj.parse_values(card='AXC')
    morph_data_obj.ayc = return_dict[morph_cards[34]] #morph_data_obj.parse_values(card='AYC')
    morph_data_obj.mxc = return_dict[morph_cards[35]] #morph_data_obj.parse_values(card='MXC')
    morph_data_obj.myc = return_dict[morph_cards[36]] #morph_data_obj.parse_values(card='MYC')


    morph_data_obj.mid1_mstat = return_dict[morph_cards[37]] #morph_data_obj.parse_values(card='MID_MP')
    morph_data_obj.mid1_ma1 = return_dict[morph_cards[38]] #morph_data_obj.parse_values(card='MID_A1')
    morph_data_obj.mid1_ma2 = return_dict[morph_cards[39]] #morph_data_obj.parse_values(card='MID_A2')
    morph_data_obj.mid1_mlev = return_dict[morph_cards[40]] #morph_data_obj.parse_values(card='MID_LEV')
    morph_data_obj.mid1_istat = return_dict[morph_cards[41]] #morph_data_obj.parse_values(card='MID_I')
    morph_data_obj.mid1_ixp = return_dict[morph_cards[42]] #morph_data_obj.parse_values(card='MID_IXP')
    morph_data_obj.mid1_iyp = return_dict[morph_cards[43]] #morph_data_obj.parse_values(card='MID_IYP')
    morph_data_obj.mid1_dstat = return_dict[morph_cards[44]] #morph_data_obj.parse_values(card='MID_D')
    morph_data_obj.mid1_darea = return_dict[morph_cards[45]] #morph_data_obj.parse_values(card='MID_DA')
    morph_data_obj.mid1_gini = return_dict[morph_cards[46]] #morph_data_obj.parse_values(card='MID_GINI')
    morph_data_obj.mid1_m20 = return_dict[morph_cards[47]] #morph_data_obj.parse_values(card='MID_M20')
    morph_data_obj.mid1_area = return_dict[morph_cards[48]] #morph_data_obj.parse_values(card='MID_AREA')
    morph_data_obj.mid1_snp = return_dict[morph_cards[49]] #morph_data_obj.parse_values(card='MID_SNP')


    morph_data_obj.mid2_mstat = return_dict[morph_cards[50]] #morph_data_obj.parse_values(card='MID2_MP')
    morph_data_obj.mid2_ma1 = return_dict[morph_cards[51]] #morph_data_obj.parse_values(card='MID2_A1')
    morph_data_obj.mid2_ma2 = return_dict[morph_cards[52]] #morph_data_obj.parse_values(card='MID2_A2')
    morph_data_obj.mid2_mlev = return_dict[morph_cards[53]] #morph_data_obj.parse_values(card='MID2_LEV')
    morph_data_obj.mid2_istat = return_dict[morph_cards[54]] #morph_data_obj.parse_values(card='MID2_I')
    morph_data_obj.mid2_ixp = return_dict[morph_cards[55]] #morph_data_obj.parse_values(card='MID2_IXP')
    morph_data_obj.mid2_iyp = return_dict[morph_cards[56]] #morph_data_obj.parse_values(card='MID2_IYP')
    morph_data_obj.mid2_dstat = return_dict[morph_cards[57]] #morph_data_obj.parse_values(card='MID2_D')
    morph_data_obj.mid2_darea = return_dict[morph_cards[58]] #morph_data_obj.parse_values(card='MID2_DA')

    #print "finished morph cards"

    #print morph_data_obj.gini.shape
    if get_idl:
        morph_data_obj.idlgini = morph_data_obj.parse_idl_values(col='col22')
        morph_data_obj.idlm20 = morph_data_obj.parse_idl_values(col='col23')
        morph_data_obj.idlcc = morph_data_obj.parse_idl_values(col='col17')
        morph_data_obj.idlrpe = morph_data_obj.parse_idl_values(col='col10')
        morph_data_obj.idlsnp = morph_data_obj.parse_idl_values(col='col6')
        morph_data_obj.idlasym = morph_data_obj.parse_idl_values(col='col20')
        morph_data_obj.idlr20 = morph_data_obj.parse_idl_values(col='col18')
        morph_data_obj.idlr80 = morph_data_obj.parse_idl_values(col='col19')
        morph_data_obj.idlelong = morph_data_obj.parse_idl_values(col='col11')
        morph_data_obj.idlaxc = morph_data_obj.parse_idl_values(col='col13')
        morph_data_obj.idlayc = morph_data_obj.parse_idl_values(col='col14')
        morph_data_obj.idlmxc = morph_data_obj.parse_idl_values(col='col15')
        morph_data_obj.idlmyc = morph_data_obj.parse_idl_values(col='col16')
        #print morph_data_obj.idlgini.shape



    #pickle object to disk
    if dofull:
        outname = 'MorphDataObject_'+depths[0]+'.pickle'
        outf = open(outname,'w')
        result = cPickle.dump(morph_data_obj,outf)
        outf.close()
        morph_data_obj_light = copy.copy(morph_data_obj)
    else:
        morph_data_obj_light = morph_data_obj

    morph_data_obj_light.morph_headers = None
    morph_data_obj_light.si_headers = None
    morph_data_obj_light.pum_headers = None
    morph_data_obj_light.value_dicts = None


    lightoutname = 'MorphDataObjectLight_'+depths[0]+'.pickle'
    outf = open(lightoutname,'w')
    result = cPickle.dump(morph_data_obj_light,outf)
    outf.close()    


    os.chdir(cwd)
    return morph_data_obj


def return_morph_object(snapshot_directory='/Users/gsnyder/Dropbox/Projects/snapshot_060',clobber=False,light=False,depth='SB25'):
    if light:
        morph_pickle_file = os.path.join(snapshot_directory,'MorphDataObjectLight_'+depth+'.pickle')
    else:
        morph_pickle_file = os.path.join(snapshot_directory,'MorphDataObject_'+depth+'.pickle')


    if not os.path.lexists(morph_pickle_file) or clobber==True:

        sh_array = None #sh_array = np.asarray([0,1,10107,10108,10301,10302,10561,
        #                10722,10723,10928,10929,11228,11397,
        #                11398,11399,11581,11582,11757,11941,
        #                11942,12,12181,12182,12383,12384,12542,
        #                       1268,14,15,1780,1781,19887,2,2332,2333,2334])
        morph_data_obj = parse_illustris_morph_snapshot(snapshot_directory=snapshot_directory,sh_array=sh_array)
    else:
        loadf = open(morph_pickle_file,'r')
        morph_data_obj = cPickle.load(loadf)
        loadf.close()

    return morph_data_obj



def compare_060_000():
    snapshot_directory='/Users/gsnyder/Dropbox/Projects/snapshot_060'

    morph_obj = return_morph_object(snapshot_directory=snapshot_directory,light=True,clobber=True)

    snpix = morph_obj.snpix


    #plot stuff
    f1 = pyplot.figure(figsize=(4.5,4.5), dpi=150)
    pyplot.subplots_adjust(left=0.11, right=0.98, bottom=0.08, top=0.99,wspace=0.25,hspace=0.25)

    skip = 25
    comparison_subplot(f1,3,2,1,'$G_{Py}-G_{IDL}$',0.10,morph_obj.gini,morph_obj.idlgini,snpix,skip=skip)
    comparison_subplot(f1,3,2,2,'$M_{20,Py}-M_{20,IDL}$',0.30,morph_obj.m20,morph_obj.idlm20,snpix,skip=skip)
    comparison_subplot(f1,3,2,3,'$C_{Py}-C_{IDL}$',0.4,morph_obj.cc,morph_obj.idlcc,snpix,skip=skip)
    comparison_subplot(f1,3,2,4,'$A_{Py}-A_{IDL}$',0.1,morph_obj.asym,morph_obj.idlasym,snpix,skip=skip)

    gi = np.where(np.logical_and(np.logical_and(morph_obj.r20 != -9.0, morph_obj.idlr20 != -9.0),snpix != -9.0))[0]
    comparison_subplot(f1,3,2,5,'$r_{20,Py}-r_{20,IDL} (arcsec)$',0.05,morph_obj.r20*morph_obj.pix_arcsec,morph_obj.idlr20*morph_obj.pix_arcsec,snpix,skip=skip,gi=gi)
    gi = np.where(np.logical_and(np.logical_and(morph_obj.rpe != -9.0, morph_obj.idlrpe != -9.0),snpix != -9.0))[0]
    comparison_subplot(f1,3,2,6,'$r_{p,Py}-r_{p,IDL} (arcsec)$',0.2,morph_obj.rpe*morph_obj.pix_arcsec,morph_obj.idlrpe*morph_obj.pix_arcsec,snpix,skip=skip,gi=gi)



    f1.savefig('CompareMorphs_IDL_Python.pdf',format='pdf')
    pyplot.close(f1)



    return


def comparison_subplot(f1,ny,nx,n,ylabel,ylim,pyval,idlval,snpix,skip=1,gi=None):
    snbins = np.linspace(0,25,20)

    axi = f1.add_subplot(ny,nx,n)
    axi.locator_params(nbins=5,prune='both')
    axi.set_ylabel(ylabel,size=7,labelpad=1)    
    axi.set_xlabel('$<S/N>_{Py}$',size=7,labelpad=1)
    axi.set_xlim(-1,25)
    axi.set_ylim(-1.0*ylim,ylim)    
    axi.tick_params(axis='both',which='major',labelsize=7)

    diff = pyval - idlval
    if gi is None:
        gi = np.where(np.logical_and(np.logical_and(pyval != -9.0, idlval != -9.0),snpix != -9.0))[0]

    diff_meds = np.zeros_like(snbins[1:])
    bin_centers = np.zeros_like(snbins[1:])
    diff_mads = np.zeros_like(snbins[1:])

    #print gi.shape

    for i,snb in enumerate(snbins[1:]):
        le = snbins[i]
        si = np.where(np.logical_and(snpix[gi] >= le,snpix[gi] < snb))[0]
        diff_meds[i] = np.median(diff[gi[si]].flatten())
        bin_centers[i] = (snb+le)/2.0
        diff_mads[i] = msbs.MAD(diff[gi[si]].flatten())

    #axi.plot(snpix[gi].flatten()[::skip], diff[gi].flatten()[::skip],'^k',markersize=0.2)
    axi.scatter(snpix[gi].flatten()[::skip], diff[gi].flatten()[::skip], c='Black', s=0.3, lw=0, edgecolor='None')

    axi.plot([0.1,50],[0.0,0.0])
    axi.plot([3,3],[-5,5])
    #axi.plot(bin_centers,diff_meds,'+r',markersize=5)
    axi.errorbar(bin_centers,diff_meds,xerr=None,yerr=diff_mads,marker='o',color='Red',linestyle='None',elinewidth=0.5,capthick=0.5,capsize=1.0,markersize=3,mew=0)
    sig_5 = np.interp(5.0,bin_centers,diff_mads)
    med_5 = np.interp(5.0,bin_centers,diff_meds)
    axi.annotate('$\sigma_5 = ${:5.4f}'.format(sig_5),(0.6,0.85),xycoords='axes fraction',size=7,color='Red')
    axi.annotate('$\Delta_5 = ${:5.4f}'.format(med_5),(0.6,0.75),xycoords='axes fraction',size=7,color='Red')

    return



def get_environment(pri_snap,pri_sfid,bp='/astro/snyder_lab2/Illustris/Illustris-1'):

    efile = os.path.join(bp,'vrg_environment/environment_{:03d}'.format(pri_snap)+'.hdf5')
    delta = None
    hsml = None

    if not os.path.lexists(efile):
        return np.zeros_like(pri_sfid) - 9.0, np.zeros_like(pri_sfid) - 9.0

    with h5py.File(efile,'r') as f:
        delta = f['delta'].value[pri_sfid]
        #hsml = f['hsml'].value[pri_sfid]

    assert delta is not None
 
    return delta #, hsml


def get_other(snapnum,sfid,bp='/astro/snyder_lab2/Illustris/Illustris-1'):
    ofile = os.path.join(bp,'stellar_circs.hdf5')
    snapname = 'Snapshot_'+str(snapnum)
    keys = [u'CircAbove07Frac', u'CircAbove07Frac_allstars', u'CircAbove07MinusBelowNeg07Frac', u'CircAbove07MinusBelowNeg07Frac_allstars', u'CircTwiceBelow0Frac', u'CircTwiceBelow0Frac_allstars', u'MassTensorEigenVals', u'ReducedMassTensorEigenVals', u'SpecificAngMom', u'SpecificAngMom_allstars', u'SubfindID']

    

    with h5py.File(ofile,'r') as f:
        SubfindID = f[snapname]['SubfindID'].value
        ouri = [] #index into SubfindIDs where we have a morph entry
        sf_list = []

        for i,sfi in enumerate(sfid):
            mi = np.where(SubfindID==sfi)[0]
            if mi.shape[0] > 0:
                ouri.append(mi[0])
                sf_list.append(i) #index into sfid

        SubfindID = SubfindID[ouri]
        CircAbove07Frac = f[snapname]['CircAbove07Frac'].value[ouri]
        CircAbove07Frac_allstars = f[snapname]['CircAbove07Frac_allstars'].value[ouri]
        CircAbove07MinusBelowNeg07Frac = f[snapname]['CircAbove07MinusBelowNeg07Frac'].value[ouri]
        CircAbove07MinusBelowNeg07Frac_allstars = f[snapname]['CircAbove07MinusBelowNeg07Frac_allstars'].value[ouri]
        CircTwiceBelow0Frac = f[snapname]['CircTwiceBelow0Frac'].value[ouri]
        CircTwiceBelow0Frac_allstars = f[snapname]['CircTwiceBelow0Frac_allstars'].value[ouri]

        if snapnum >= 38:
            MassTensorEigenVals = f[snapname]['MassTensorEigenVals'].value[ouri]
            ReducedMassTensorEigenVals = f[snapname]['ReducedMassTensorEigenVals'].value[ouri]


        SpecificAngMom = f[snapname]['SpecificAngMom'].value[ouri]
        SpecificAngMom_allstars = f[snapname]['SpecificAngMom_allstars'].value[ouri]

    #print sfid[sf_list]
    #print SubfindID

    assert np.all(sfid[sf_list]==SubfindID)

    if snapnum >= 38:
        outdict = {'SubfindID':SubfindID,'CircAbove07Frac':CircAbove07Frac,'CircAbove07Frac_allstars':CircAbove07Frac_allstars,
                   'CircAbove07MinusBelowNeg07Frac':CircAbove07MinusBelowNeg07Frac,'CircAbove07MinusBelowNeg07Frac_allstars':CircAbove07MinusBelowNeg07Frac_allstars,
                   'CircTwiceBelow0Frac':CircTwiceBelow0Frac,'CircTwiceBelow0Frac_allstars':CircTwiceBelow0Frac_allstars,'MassTensorEigenVals':MassTensorEigenVals,
                   'ReducedMassTensorEigenVals':ReducedMassTensorEigenVals,'SpecificAngMom':SpecificAngMom,'SpecificAngMom_allstars':SpecificAngMom_allstars}
    else:
        outdict = {'SubfindID':SubfindID,'CircAbove07Frac':CircAbove07Frac,'CircAbove07Frac_allstars':CircAbove07Frac_allstars,
                   'CircAbove07MinusBelowNeg07Frac':CircAbove07MinusBelowNeg07Frac,'CircAbove07MinusBelowNeg07Frac_allstars':CircAbove07MinusBelowNeg07Frac_allstars,
                   'CircTwiceBelow0Frac':CircTwiceBelow0Frac,'CircTwiceBelow0Frac_allstars':CircTwiceBelow0Frac_allstars,'SpecificAngMom':SpecificAngMom,'SpecificAngMom_allstars':SpecificAngMom_allstars}
        
    return outdict, sf_list



def load_all_info(base_directory='/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/',snapnum=68, filename = 'MorphDataObjectLight_SB25.pickle'):

    snapdir = os.path.join(base_directory,'snapshot_{:03d}'.format(snapnum))
    morphcat = os.path.join(snapdir,filename)

    print("loading.. ", morphcat)

    loadf = open(morphcat,'r')
    morph_data_obj = cPickle.load(loadf)
    loadf.close()

    sfids = np.int64(morph_data_obj.unique_subhalo_ids)

    subhalo_dict = ilpy.groupcat.loadSubhalos('/astro/snyder_lab2/Illustris/Illustris-1',snapnum,fields=shfields)
    newsubdict = {}
    for f in subhalo_dict:
        subfield = subhalo_dict[f]
        if f=='count':
            newsubdict[f]=subfield
            continue
        newsubdict[f] = subfield[sfids]

    
    otherdict,sf_list = get_other(int(snapnum),sfids)
    newdict = {}
    for f in otherdict.keys():
        otherfield = otherdict[f]
        newfield = np.zeros_like(sfids,dtype=otherfield.dtype) - 9
        newfield[sf_list]= otherfield
        newdict[f]=newfield


    delta = get_environment(int(snapnum), sfids)


    return morph_data_obj, subhalo_dict, newsubdict, delta, otherdict, newdict


def load_everything(base_directory='/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/',
                    snaps = ['snapshot_035','snapshot_038','snapshot_041','snapshot_045','snapshot_049','snapshot_054','snapshot_060','snapshot_064','snapshot_068','snapshot_075','snapshot_085','snapshot_103'],
                    filename = 'MorphDataObjectLight_SB25.pickle'):

    morphobjs = {}
    dataobjs = {} ; dataobjs['morph']={} ; dataobjs['subhalos']={} ; dataobjs['delta']={} ; dataobjs['other']={}

    for s in snaps:
        morph_data_obj, subhalo_dict, newsubdict, delta, otherdict, newdict = load_all_info(base_directory=base_directory,snapnum = int(s[-3:]),filename=filename)

        dataobjs['morph'][s]=morph_data_obj
        dataobjs['subhalos'][s]=newsubdict
        dataobjs['delta'][s]=delta
        dataobjs['other'][s]=newdict

    return dataobjs,snaps


def output_full_catalog(dataobjs,snaps,filter=['WFC3-F160W'],label='SB25',camera=[0,1,2,3],basic=True):


    #dataobjs,snaps = load_everything(base_directory=base_directory,filename=filename)
    #gini,m20,cc,rpe,rhc,snp,sfid

    
    output_dataset = {}
    for s in snaps:
        output_dataset[s] = {}
        output_dataset[s]['GINI']=dataobjs['morph'][s].gini[:,:,:,0]
        output_dataset[s]['M20']=dataobjs['morph'][s].m20[:,:,:,0]
        output_dataset[s]['CC']=dataobjs['morph'][s].cc[:,:,:,0]
        output_dataset[s]['RP']=dataobjs['morph'][s].rpe[:,:,:,0]
        output_dataset[s]['RHALF']=dataobjs['morph'][s].rhalfe[:,:,:,0]
        output_dataset[s]['SNPIX']=dataobjs['morph'][s].snpix[:,:,:,0]
        output_dataset[s]['MAG']=dataobjs['morph'][s].magseg[:,:,:,0]
        output_dataset[s]['MAG_ERR']=dataobjs['morph'][s].magseg_err[:,:,:,0]
        output_dataset[s]['SEG_AREA']=dataobjs['morph'][s].seg_area[:,:,:,0]
        output_dataset[s]['PIX_ARCSEC']=dataobjs['morph'][s].pix_arcsec[:,:,:,0]

        if basic is False:
            output_dataset[s]['ASYM']=dataobjs['morph'][s].asym[:,:,:,0]
            output_dataset[s]['MID1_MPRIME']=dataobjs['morph'][s].mid1_mstat[:,:,:,0]
            output_dataset[s]['MID1_ISTAT']=dataobjs['morph'][s].mid1_istat[:,:,:,0]
            output_dataset[s]['MID1_DSTAT']=dataobjs['morph'][s].mid1_dstat[:,:,:,0]
            output_dataset[s]['MID1_A1']=dataobjs['morph'][s].mid1_ma1[:,:,:,0]
            output_dataset[s]['MID1_A2']=dataobjs['morph'][s].mid1_ma2[:,:,:,0]
            output_dataset[s]['MID1_AREA']=dataobjs['morph'][s].mid1_area[:,:,:,0]
            output_dataset[s]['MID1_GINI']=dataobjs['morph'][s].mid1_gini[:,:,:,0]
            output_dataset[s]['MID1_M20']=dataobjs['morph'][s].mid1_m20[:,:,:,0]
            output_dataset[s]['MID1_SNP']=dataobjs['morph'][s].mid1_snp[:,:,:,0]

            output_dataset[s]['MID2_MPRIME']=dataobjs['morph'][s].mid1_mstat[:,:,:,0]
            output_dataset[s]['MID2_ISTAT']=dataobjs['morph'][s].mid1_istat[:,:,:,0]
            output_dataset[s]['MID2_DSTAT']=dataobjs['morph'][s].mid1_dstat[:,:,:,0]
            output_dataset[s]['MID2_A1']=dataobjs['morph'][s].mid1_ma1[:,:,:,0]
            output_dataset[s]['MID2_A2']=dataobjs['morph'][s].mid1_ma2[:,:,:,0]

            output_dataset[s]['ELONG']=dataobjs['morph'][s].elong[:,:,:,0]
            output_dataset[s]['ORIENT']=dataobjs['morph'][s].orient[:,:,:,0]
            output_dataset[s]['SEG_ELLIPT']=dataobjs['morph'][s].seg_ellipt[:,:,:,0]
            output_dataset[s]['SEG_ECCENT']=dataobjs['morph'][s].seg_eccent[:,:,:,0]
            output_dataset[s]['SEG_EQRAD']=dataobjs['morph'][s].seg_eqrad[:,:,:,0]
            output_dataset[s]['SEG_SMAJSIG']=dataobjs['morph'][s].seg_smajsig[:,:,:,0]
            output_dataset[s]['SEG_SMINSIG']=dataobjs['morph'][s].seg_sminsig[:,:,:,0]

            output_dataset[s]['M_A']=dataobjs['morph'][s].m_a[:,:,:,0]
            output_dataset[s]['M_I2']=dataobjs['morph'][s].m_i2[:,:,:,0]

            output_dataset[s]['FLAG']=dataobjs['morph'][s].flag[:,:,:,0]
            output_dataset[s]['CFLAG']=dataobjs['morph'][s].cflag[:,:,:,0]
            output_dataset[s]['CC_ERR']=dataobjs['morph'][s].cc_err[:,:,:,0]
            output_dataset[s]['RP_ERR']=dataobjs['morph'][s].rpe_err[:,:,:,0]
            output_dataset[s]['RHALF_ERR']=dataobjs['morph'][s].rhalfe_err[:,:,:,0]
            output_dataset[s]['AXC']=dataobjs['morph'][s].axc[:,:,:,0]
            output_dataset[s]['AYC']=dataobjs['morph'][s].ayc[:,:,:,0]

            output_dataset[s]['SUNRISE_ABSMAG']=dataobjs['morph'][s].sunrise_absmag[:,:,:,0]
            output_dataset[s]['RMS']=dataobjs['morph'][s].rms[:,:,:,0]
            output_dataset[s]['APPROXPSF_ARCSEC']=dataobjs['morph'][s].approx_psf_fwhm_arcsec[:,:,:,0]
            output_dataset[s]['REDSHIFT']=dataobjs['morph'][s].redshift[:,:,:,0]

            i=dataobjs['morph'][s].unique_subhalo_inverse
            j=dataobjs['morph'][s].cam_indices
            k=dataobjs['morph'][s].fil_indices

            imfiles = np.empty_like(dataobjs['morph'][s].gini[:,:,:,0],dtype='|S100')
            imfiles[i,j,k]=dataobjs['morph'][s].image_files
            output_dataset[s]['IMFILES']=imfiles
            
    outfile = 'nonparmorphs_'+label+'.hdf5'

    with h5py.File(outfile,'w') as writefile:
        grp = writefile.create_group('nonparmorphs')
        for s in snaps:
            sgrp = grp.create_group(s)
            sgrp.create_dataset('SubfindID',data=np.int64(dataobjs['morph'][s].unique_subhalo_ids))
            sgrp.create_dataset('Filters',data=filter)
            sgrp.create_dataset('Cameras',data=camera)
            mstar_1 = dataobjs['subhalos'][s]['SubhaloMassInRadType'][:,4].flatten()*(1.0e10)/ilh
            sfr_1 = dataobjs['subhalos'][s]['SubhaloSFR'][:].flatten()
            mhalo_1 = dataobjs['subhalos'][s]['SubhaloMass'][:].flatten()*(1.0e10)/ilh
            bhrate_1 = dataobjs['subhalos'][s]['SubhaloBHMdot'][:].flatten()*((1.0e10)/ilh)/(0.978*1.0e9/ilh)
            bhmass_1 = dataobjs['subhalos'][s]['SubhaloBHMass'][:].flatten()*(1.0e10)/ilh 

            sgrp.create_dataset('Mstar_Msun',data=mstar_1)
            sgrp.create_dataset('SFR_Msunperyr',data=sfr_1)
            sgrp.create_dataset('BHMdot_Msunperyr',data=bhrate_1)
            sgrp.create_dataset('Mbh_Msun',data=bhmass_1)
            sgrp.create_dataset('Mhalo_Msun',data=mhalo_1)
            
            for fil in filter:
                fsgrp = sgrp.create_group(fil)
                fi = np.where(dataobjs['morph'][s].filters==fil)[0]

                for c in camera:
                    cname = 'CAMERA'+str(c)
                    cfsgrp = fsgrp.create_group(cname)

                    for field in output_dataset[s].keys():
                        this_array = output_dataset[s][field][:,c,fi].flatten()
                        if field != 'IMFILES':
                            nan_values = np.where(this_array == -9.0,np.nan*np.ones_like(this_array),this_array)
                        if field=='MAG' or field=='MAG_ERR':
                            nan_values = np.where(nan_values==-1.0,np.nan*np.ones_like(nan_values),nan_values)

                        if field=='IMFILES':
                            dset = cfsgrp.create_dataset(field,data=this_array)
                        else:
                            dset = cfsgrp.create_dataset(field,data=nan_values)

    return

def output_catalog(base_directory='/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/',snapnum='068', filename = 'MorphDataObjectLight_SB25.pickle',filter='HST-F160W',label='SB25'):

    morph_data_obj, subhalo_dict, newsubdict, delta, hsml, otherdict, newdict = load_all_info(base_directory=base_directory,snapnum=snapnum,filename=filename)

    #write HDF5 file
    exampleset = subhalo_dict['SubhaloMass']
    gini_0 = np.zeros_like(exampleset)
    gini_1 = np.zeros_like(exampleset)
    gini_2 = np.zeros_like(exampleset)
    gini_3 = np.zeros_like(exampleset)

    m20_0 = np.zeros_like(exampleset)
    m20_1 = np.zeros_like(exampleset)
    m20_2 = np.zeros_like(exampleset)
    m20_3 = np.zeros_like(exampleset)

    cc_0 = np.zeros_like(exampleset)
    cc_1 = np.zeros_like(exampleset)
    cc_2 = np.zeros_like(exampleset)
    cc_3 = np.zeros_like(exampleset)

    rpe_0 = np.zeros_like(exampleset)
    rpe_1 = np.zeros_like(exampleset)
    rpe_2 = np.zeros_like(exampleset)
    rpe_3 = np.zeros_like(exampleset)

    rhalfe_0 = np.zeros_like(exampleset)
    rhalfe_1 = np.zeros_like(exampleset)
    rhalfe_2 = np.zeros_like(exampleset)
    rhalfe_3 = np.zeros_like(exampleset)

    snp_0 = np.zeros_like(exampleset)
    snp_1 = np.zeros_like(exampleset)
    snp_2 = np.zeros_like(exampleset)
    snp_3 = np.zeros_like(exampleset)

    sfids = np.int64(morph_data_obj.unique_subhalo_ids)

    fi = np.where(morph_data_obj.filters==filter)[0]

    gini_0 = (morph_data_obj.gini)[:,0,fi,0].flatten()
    gini_1 = (morph_data_obj.gini)[:,1,fi,0].flatten()
    gini_2 = (morph_data_obj.gini)[:,2,fi,0].flatten()
    gini_3 = (morph_data_obj.gini)[:,3,fi,0].flatten()
    
    m20_0 = (morph_data_obj.m20)[:,0,fi,0].flatten()
    m20_1 = (morph_data_obj.m20)[:,1,fi,0].flatten()
    m20_2 = (morph_data_obj.m20)[:,2,fi,0].flatten()
    m20_3 = (morph_data_obj.m20)[:,3,fi,0].flatten()
    
    cc_0 = (morph_data_obj.cc)[:,0,fi,0].flatten()
    cc_1 = (morph_data_obj.cc)[:,1,fi,0].flatten()
    cc_2 = (morph_data_obj.cc)[:,2,fi,0].flatten()
    cc_3 = (morph_data_obj.cc)[:,3,fi,0].flatten()
    
    rpe_0 = (morph_data_obj.rpe)[:,0,fi,0].flatten()
    rpe_1 = (morph_data_obj.rpe)[:,1,fi,0].flatten()
    rpe_2 = (morph_data_obj.rpe)[:,2,fi,0].flatten()
    rpe_3 = (morph_data_obj.rpe)[:,3,fi,0].flatten()
    
    rhalfe_0 = (morph_data_obj.rhalfe)[:,0,fi,0].flatten()
    rhalfe_1 = (morph_data_obj.rhalfe)[:,1,fi,0].flatten()
    rhalfe_2 = (morph_data_obj.rhalfe)[:,2,fi,0].flatten()
    rhalfe_3 = (morph_data_obj.rhalfe)[:,3,fi,0].flatten()
    
    snp_0 = (morph_data_obj.snpix)[:,0,fi,0].flatten()
    snp_1 = (morph_data_obj.snpix)[:,1,fi,0].flatten()
    snp_2 = (morph_data_obj.snpix)[:,2,fi,0].flatten()
    snp_3 = (morph_data_obj.snpix)[:,3,fi,0].flatten()
    

    outfile = 'nonparmorphs_'+label+'_'+filter+'_{:03d}'.format(snapnum)+'.hdf5'
    writefile = h5py.File(outfile,'w')
    dset = writefile.create_dataset('SubfindID',data=np.int64(sfids)) ; dset.attrs['blame']='gsnyder@stsci.edu' ; dset.attrs['ref']='HST13887' ; dset.attrs['depthmagsqas']=morph_data_obj.depths[0][-2:] ; dset.attrs['filter']=filter
    dset = writefile.create_dataset('GINI_cam0',data=np.float32(gini_0)) ; dset.attrs['info']='Gini,Cam 0'
    dset = writefile.create_dataset('GINI_cam1',data=np.float32(gini_1))
    dset = writefile.create_dataset('GINI_cam2',data=np.float32(gini_2))
    dset = writefile.create_dataset('GINI_cam3',data=np.float32(gini_3))
    dset = writefile.create_dataset('M20_cam0',data=np.float32(m20_0)) ; dset.attrs['info']='M20,Cam 0'
    dset = writefile.create_dataset('M20_cam1',data=np.float32(m20_1))
    dset = writefile.create_dataset('M20_cam2',data=np.float32(m20_2))
    dset = writefile.create_dataset('M20_cam3',data=np.float32(m20_3))
    dset = writefile.create_dataset('CONCENTRATION_cam0',data=np.float32(cc_0)) ; dset.attrs['info']='Concentration,Cam 0'
    dset = writefile.create_dataset('CONCENTRATION_cam1',data=np.float32(cc_1))
    dset = writefile.create_dataset('CONCENTRATION_cam2',data=np.float32(cc_2))
    dset = writefile.create_dataset('CONCENTRATION_cam3',data=np.float32(cc_3))
    dset = writefile.create_dataset('RP_cam0',data=np.float32(rpe_0)) ; dset.attrs['info']='PetrosianRadius,Cam 0'
    dset = writefile.create_dataset('RP_cam1',data=np.float32(rpe_1))
    dset = writefile.create_dataset('RP_cam2',data=np.float32(rpe_2))
    dset = writefile.create_dataset('RP_cam3',data=np.float32(rpe_3))
    dset = writefile.create_dataset('RHALF_cam0',data=np.float32(rhalfe_0)) ; dset.attrs['info']='HalfLightRadius,Cam 0'
    dset = writefile.create_dataset('RHALF_cam1',data=np.float32(rhalfe_1))
    dset = writefile.create_dataset('RHALF_cam2',data=np.float32(rhalfe_2))
    dset = writefile.create_dataset('RHALF_cam3',data=np.float32(rhalfe_3))
    dset = writefile.create_dataset('SN_cam0',data=np.float32(snp_0)) ; dset.attrs['info']='avgSNratio_perPixel,Cam 0'
    dset = writefile.create_dataset('SN_cam1',data=np.float32(snp_1))
    dset = writefile.create_dataset('SN_cam2',data=np.float32(snp_2))
    dset = writefile.create_dataset('SN_cam3',data=np.float32(snp_3))

    writefile.close()

    #write ascii file e.g., Snyder 2015 
    ### Non-parametric morphology catalog
    ### gfs_morphs_nodust_g.dat
    ### February 25, 2015
    ### We kindly ask that you cite the following papers if you use this catalog in a publication: 
    ### Underlying data: Illustris Simulation, Vogelsberger et al. (2014a,b), Genel et al. (2014) 
    ### Mock image generation and optical morphologies: Snyder et al. (2015), Torrey et al. (2015), Lotz et al. (2004) 
    ###################
    ### Column 01: Illustris Simulation subhalo index for snapshot 135 (z=0)
    ### Column 02: Log10 of stellar mass within 2x stellar half-mass radius [fiducial M_* from Vogelsberger et al. 2014], solar units
    ### Column 03: Star formation rate in subhalo, in Msun/year 
    ### Column 04: Intrinsic U-B color (AB system, Johnson filters) of subhalo, dust-free 
    ### Column 05: Camera number from 2013 Sunrise run 
    ### Column 06: Gini, following Lotz et al. 2004
    ### Column 07: M_20, following Lotz et al. 2004
    ### Column 08: Concentration, following Lotz et al. 2004
    ### Column 09: elliptical Petrosian radius, in kpc, following Lotz et al. 2004
    ### Column 10: elliptical half-light radius, in kpc, following Lotz et al. 2004
    ###################



    return



def make_master_catalog(base_directory='/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/',depth='SB25'):
    morph_data_obj = parse_illustris_morph_snapshot(snapshot_directory=os.path.join(base_directory,'snapshot_035'),depth=depth)
    morph_data_obj = parse_illustris_morph_snapshot(snapshot_directory=os.path.join(base_directory,'snapshot_038'),depth=depth)
    morph_data_obj = parse_illustris_morph_snapshot(snapshot_directory=os.path.join(base_directory,'snapshot_041'),depth=depth)
    morph_data_obj = parse_illustris_morph_snapshot(snapshot_directory=os.path.join(base_directory,'snapshot_045'),depth=depth)
    morph_data_obj = parse_illustris_morph_snapshot(snapshot_directory=os.path.join(base_directory,'snapshot_049'),depth=depth)
    morph_data_obj = parse_illustris_morph_snapshot(snapshot_directory=os.path.join(base_directory,'snapshot_054'),depth=depth)
    morph_data_obj = parse_illustris_morph_snapshot(snapshot_directory=os.path.join(base_directory,'snapshot_060'),depth=depth)
    morph_data_obj = parse_illustris_morph_snapshot(snapshot_directory=os.path.join(base_directory,'snapshot_064'),depth=depth)
    morph_data_obj = parse_illustris_morph_snapshot(snapshot_directory=os.path.join(base_directory,'snapshot_068'),depth=depth)
    morph_data_obj = parse_illustris_morph_snapshot(snapshot_directory=os.path.join(base_directory,'snapshot_075'),depth=depth)
    morph_data_obj = parse_illustris_morph_snapshot(snapshot_directory=os.path.join(base_directory,'snapshot_085'),depth=depth)
    morph_data_obj = parse_illustris_morph_snapshot(snapshot_directory=os.path.join(base_directory,'snapshot_103'),depth=depth)

    return 1




ilh = 0.704
illcos = astropy.cosmology.FlatLambdaCDM(H0=70.4,Om0=0.2726,Ob0=0.0456)

shfields=['SubhaloBHMass','SubhaloBHMdot','SubhaloGrNr','SubhaloMass','SubhaloMassInRadType','SubhaloMassType','SubhaloParent','SubhaloPos','SubhaloSFR','SubhaloSFRinRad','SubhaloSpin','SubhaloStarMetallicity','SubhaloVel','SubhaloVelDisp','SubhaloVmax', 'SubhaloVmaxRad']



if __name__=="__main__":
    #compare_060_000()
    #make_master_catalog(depth='SB25')
    #make_master_catalog(depth='SB27')
    #make_master_catalog(depth='SB29')
    base_directory='/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/'
    morph_data_obj = parse_illustris_morph_snapshot(snapshot_directory=os.path.join(base_directory,'snapshot_068'),depth='SB27')
