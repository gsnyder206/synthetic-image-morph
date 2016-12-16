#import ez_galaxy
import math
import string
import sys
import struct
import matplotlib
import matplotlib.pyplot as pyplot
#import utils
import numpy as np
#import array
#import astLib.astStats as astStats
#import cPickle
import scipy.ndimage
import scipy.stats as ss
import scipy as sp
import astropy.io.fits as pyfits
import glob
import os
import gc

def little_f(x, minx, maxx,Q,alph):
	
	f = sp.zeros_like(x)
	
	
	maxx = minx + sp.sinh(Q)/(alph*Q) ; #print maxx
	f = sp.arcsinh(alph*Q*(x-minx))/Q
	f[sp.where(x < minx)]=0.0
	f[sp.where(x > maxx)]=1.0
	
	return f







def make_nasa(b,g,r,filename,alph,Q,inches=5.0,dpi=72,fwhm_pixels=[0.0,0.0,0.0],sigma_tuple=[0.0,0.0,0.0],zlabel=-1, use_inches=False):

	#fpar = open(filename+'-rgbparams.txt','w')
	#fpar.write(filename+'\n')
	#fpar.write('alph= {:10e}, Q= {:10e}, inches= {:12.4f}, dpi= {:04d}, fwhm_pixels= {:12.4f}'.format(alph,Q,inches,dpi,fwhm_pixels)+'\n')
	#fpar.close()
	
	b = b*1.0
	g = g*1.0
	r = r*1.0

	if fwhm_pixels[0] > 1.0e-5:
		sR=np.zeros_like(r) ; sG = np.zeros_like(g) ; sB = np.zeros_like(b)
		fwhm = fwhm_pixels #pixels, 0.5kpc/pixel
		sigma = fwhm/(2.0*math.sqrt(2.0*math.log(2.0)))
		resR = sp.ndimage.filters.gaussian_filter(r,sigma[0],output=sR)
		resG = sp.ndimage.filters.gaussian_filter(g,sigma[1],output=sG)
		resB = sp.ndimage.filters.gaussian_filter(b,sigma[2],output=sB)

		b = sB
		g = sG
		r = sR

	#I think the idea is to add sky shot noise *here*, after the sources have been convolved?  YES!  I used to know things
	if sigma_tuple[0] > 1.0e-8:
		b = b + sigma_tuple[0]*np.random.standard_normal(b.shape)

	if sigma_tuple[1] > 1.0e-8:
		g = g + sigma_tuple[1]*np.random.standard_normal(g.shape)

	if sigma_tuple[2] > 1.0e-8:
		r = r + sigma_tuple[2]*np.random.standard_normal(r.shape)

	b[sp.where(b <= 0.0)]=0.0 ; g[sp.where(g <= 0.0)]=0.0 ; r[sp.where(r <= 0.0)]=0.0


	
	I = (b+g+r)/3.0 + 1.0e-20
	
	
	minval = 0.0
	maxval = np.max(I)
	#print maxval
	
	#alph= 1.0  #fix alph to set the intensity of the faint features
	#Q=1.0  #1.0e-12 # 5.0  #e-12 #1.0e-10  #
	
	#10.0    #1.0e-10
	
	
	#factor = 1.0 #little_f(I,minval,maxval,Q,alph)/I



	
	#R = r*factor
	#G = g*factor
	#B = b*factor
	R = little_f(r,minval,maxval,Q,alph)
	G = little_f(g,minval,maxval,Q,alph)
	B = little_f(b,minval,maxval,Q,alph)
	
	imarray = np.asarray([R,G,B])
	#print imarray.shape
	
	maxrgbval = np.amax(imarray, axis=0)
	#print maxrgbval.shape
	
	changeind = np.where(maxrgbval > 1.0)
	R[changeind] = R[changeind]/maxrgbval[changeind]
	G[changeind] = G[changeind]/maxrgbval[changeind]
	B[changeind] = B[changeind]/maxrgbval[changeind]
	
	#if maxrgb > 1.0:
	#	R=R/maxrgb
	#	G=G/maxrgb
	#	B=B/maxrgb
		
		
	ind = sp.where(I < 1.0e-10)
	R[ind]=0.0 ; G[ind]=0.0 ; B[ind]=0.0


	#imarray = np.asarray(np.transpose([sR,sG,sB]))
	imarray = np.asarray(np.transpose([R,G,B]))

	leny = float( len(R[0,:]))
	lenx = float( len(R[:,0]))
	
	inx = lenx/float(dpi)
	iny = leny/float(dpi)
	#print inx, iny
	
	if use_inches==True:
		inx = inches
		iny = inches

	#print inx, iny

	f1 = pyplot.figure(figsize=( inx, iny ), dpi=dpi, frameon=False)
	pyplot.subplots_adjust(bottom=0.0, top=1.0, left=0.0, right=1.0, hspace=0.0, wspace=0.0)
	#axi = pyplot.axes()
	#axi.set_xlim(0.5, 2.0)
	#axi.set_ylim(0.70, 1.40)
	#axi.locator_params(nbins=6,prune='both')
	#ax = axes([0,0,1,1], frameon=False)
	#ax.set_axis_off()

	axi=pyplot.axes([0.0,0.0,1.0,1.0], frameon=False)
	axi.set_axis_off()
	axi.imshow(imarray[:,:,:],aspect='auto',interpolation='Nearest')
	#axi.get_xaxis().set_visible(False)
	#axi.get_yaxis().set_visible(False)
	#axi.set_frame_on(False)
	if zlabel != -1:
		axi.annotate(str(zlabel),[0.7,0.9])
	#axi.set_xlabel('Observed Redshift')
	#axi.set_ylabel('$(U-V)_0$')
	#print dpi
	
	f1.savefig(filename, dpi=dpi, format='pdf', pad_inches=0)

	pyplot.close(f1)


	
	return imarray







def make_general(b,g,r,filename,alph,Q,inches=5.0,dpi=72,fwhm_pixels=0.0,sigma_tuple=[0.0,0.0,0.0],zlabel=-1, use_inches=False):

	fpar = open(filename+'-rgbparams.txt','w')
	fpar.write(filename+'\n')
	fpar.write('alph= {:10e}, Q= {:10e}, inches= {:12.4f}, dpi= {:04d}, fwhm_pixels= {:12.4f}'.format(alph,Q,inches,dpi,fwhm_pixels)+'\n')
	fpar.close()
	
	b = b*1.0
	g = g*1.0
	r = r*1.0

	if fwhm_pixels > 1.0e-5:
		sR=np.zeros_like(r) ; sG = np.zeros_like(g) ; sB = np.zeros_like(b)
		fwhm = fwhm_pixels #pixels, 0.5kpc/pixel
		sigma = fwhm/(2.0*math.sqrt(2.0*math.log(2.0)))
		resR = sp.ndimage.filters.gaussian_filter(r,sigma,output=sR)
		resG = sp.ndimage.filters.gaussian_filter(g,sigma,output=sG)
		resB = sp.ndimage.filters.gaussian_filter(b,sigma,output=sB)

		b = sB
		g = sG
		r = sR

	#I think the idea is to add sky shot noise *here*, after the sources have been convolved?
	if sigma_tuple[0] > 1.0e-8:
		b = b + sigma_tuple[0]*np.random.standard_normal(b.shape)

	if sigma_tuple[1] > 1.0e-8:
		g = g + sigma_tuple[1]*np.random.standard_normal(g.shape)

	if sigma_tuple[2] > 1.0e-8:
		r = r + sigma_tuple[2]*np.random.standard_normal(r.shape)

	b[sp.where(b <= 0.0)]=0.0 ; g[sp.where(g <= 0.0)]=0.0 ; r[sp.where(r <= 0.0)]=0.0
	
	I = (b+g+r)/3.0 + 1.0e-20
	
	
	minval = 0.0
	maxval = np.max(I)
	#print maxval
	
	#alph= 1.0  #fix alph to set the intensity of the faint features
	#Q=1.0  #1.0e-12 # 5.0  #e-12 #1.0e-10  #
	
	#10.0    #1.0e-10
	
	
	factor = little_f(I,minval,maxval,Q,alph)/I
	
	R = r*factor
	G = g*factor
	B = b*factor
	#R = little_f(r,minval,maxval,Q,alph)
	#G = little_f(g,minval,maxval,Q,alph)
	#B = little_f(b,minval,maxval,Q,alph)
	
	
	imarray = np.asarray([R,G,B])
	#print imarray.shape
	
	maxrgbval = np.amax(imarray, axis=0)
	#print maxrgbval.shape
	
	changeind = np.where(maxrgbval > 1.0)
	R[changeind] = R[changeind]/maxrgbval[changeind]
	G[changeind] = G[changeind]/maxrgbval[changeind]
	B[changeind] = B[changeind]/maxrgbval[changeind]
	
	#if maxrgb > 1.0:
	#	R=R/maxrgb
	#	G=G/maxrgb
	#	B=B/maxrgb
		
		
	ind = sp.where(I < 1.0e-10)
	R[ind]=0.0 ; G[ind]=0.0 ; B[ind]=0.0


	#imarray = np.asarray(np.transpose([sR,sG,sB]))
	imarray = np.asarray(np.transpose([R,G,B]))

	leny = float( len(R[0,:]))
	lenx = float( len(R[:,0]))
	inx = lenx/float(dpi)
	iny = leny/float(dpi)

	if use_inches==True:
		inx = inches
		iny = inches


	
	f1 = pyplot.figure(figsize=( inx, iny ), dpi=dpi, frameon=False)
	pyplot.subplots_adjust(bottom=0.0, top=1.0, left=0.0, right=1.0, hspace=0.0, wspace=0.0)
	#axi = pyplot.axes()
	#axi.set_xlim(0.5, 2.0)
	#axi.set_ylim(0.70, 1.40)
	#axi.locator_params(nbins=6,prune='both')
	#ax = axes([0,0,1,1], frameon=False)
	#ax.set_axis_off()

	axi=pyplot.axes([0.0,0.0,1.0,1.0], frameon=False)
	axi.set_axis_off()
	axi.imshow(imarray[:,:,:],aspect='auto',interpolation='nearest')
	#axi.get_xaxis().set_visible(False)
	#axi.get_yaxis().set_visible(False)
	#axi.set_frame_on(False)
	if zlabel != -1:
		axi.annotate(str(zlabel),[0.7,0.9])
	#axi.set_xlabel('Observed Redshift')
	#axi.set_ylabel('$(U-V)_0$')
	
	f1.savefig(filename, dpi=dpi, format='png', pad_inches=0)

	pyplot.close(f1)


	
	return imarray












def make_interactive(b,g,r,alph,Q,inches=5.0,dpi=72,fwhm_pixels=0.0,sigma_tuple=[0.0,0.0,0.0],zlabel=-1):

	#fpar = open(filename+'-rgbparams.txt','w')
	#fpar.write(filename+'\n')
	#fpar.write('alph= {:10e}, Q= {:10e}, inches= {:12.4f}, dpi= {:04d}, fwhm_pixels= {:12.4f}'.format(alph,Q,inches,dpi,fwhm_pixels)+'\n')
	#fpar.close()
	
	b = b*1.0
	g = g*1.0
	r = r*1.0

	if fwhm_pixels > 1.0e-5:
		sR=np.zeros_like(r) ; sG = np.zeros_like(g) ; sB = np.zeros_like(b)
		fwhm = fwhm_pixels #pixels, 0.5kpc/pixel
		sigma = fwhm/(2.0*math.sqrt(2.0*math.log(2.0)))
		resR = sp.ndimage.filters.gaussian_filter(r,sigma,output=sR)
		resG = sp.ndimage.filters.gaussian_filter(g,sigma,output=sG)
		resB = sp.ndimage.filters.gaussian_filter(b,sigma,output=sB)

		b = sB
		g = sG
		r = sR

	#I think the idea is to add sky shot noise *here*, after the sources have been convolved?
	if sigma_tuple[0] > 1.0e-8:
		b = b + sigma_tuple[0]*np.random.standard_normal(b.shape)

	if sigma_tuple[1] > 1.0e-8:
		g = g + sigma_tuple[1]*np.random.standard_normal(g.shape)

	if sigma_tuple[2] > 1.0e-8:
		r = r + sigma_tuple[2]*np.random.standard_normal(r.shape)

	b[sp.where(b <= 0.0)]=0.0 ; g[sp.where(g <= 0.0)]=0.0 ; r[sp.where(r <= 0.0)]=0.0
	
	I = (b+g+r)/3.0 + 1.0e-20
	
	
	minval = 0.0
	maxval = np.max(I)
	#print maxval
	
	#alph= 1.0  #fix alph to set the intensity of the faint features
	#Q=1.0  #1.0e-12 # 5.0  #e-12 #1.0e-10  #
	
	#10.0    #1.0e-10
	
	
	factor = little_f(I,minval,maxval,Q,alph)/I
	
	R = r*factor
	G = g*factor
	B = b*factor
	
	
	imarray = np.asarray([R,G,B])
	#print imarray.shape
	
	maxrgbval = np.amax(imarray, axis=0)
	#print maxrgbval.shape
	
	changeind = np.where(maxrgbval > 1.0)
	R[changeind] = R[changeind]/maxrgbval[changeind]
	G[changeind] = G[changeind]/maxrgbval[changeind]
	B[changeind] = B[changeind]/maxrgbval[changeind]
	
	#if maxrgb > 1.0:
	#	R=R/maxrgb
	#	G=G/maxrgb
	#	B=B/maxrgb
		
		
	ind = sp.where(I < 1.0e-10)
	R[ind]=0.0 ; G[ind]=0.0 ; B[ind]=0.0


	#imarray = np.asarray(np.transpose([sR,sG,sB]))
	imarray = np.asarray(np.transpose([R,G,B]))

	leny = float( len(R[0,:]))
	lenx = float( len(R[:,0]))
	inx = lenx/float(dpi)
	iny = leny/float(dpi)
	#print inx, iny
	
	return imarray


def make_interactive_nasa(b,g,r,alph,Q,inches=5.0,dpi=72,fwhm_pixels=[0.0,0.0,0.0],sigma_tuple=[0.0,0.0,0.0],zlabel=-1):

	#fpar = open(filename+'-rgbparams.txt','w')
	#fpar.write(filename+'\n')
	#fpar.write('alph= {:10e}, Q= {:10e}, inches= {:12.4f}, dpi= {:04d}, fwhm_pixels= {:12.4f}'.format(alph,Q,inches,dpi,fwhm_pixels)+'\n')
	#fpar.close()
	
	b = b*1.0
	g = g*1.0
	r = r*1.0

	if fwhm_pixels[0] > 1.0e-5:
		sR=np.zeros_like(r) ; sG = np.zeros_like(g) ; sB = np.zeros_like(b)
		fwhm = fwhm_pixels #pixels, 0.5kpc/pixel
		sigma = fwhm/(2.0*math.sqrt(2.0*math.log(2.0)))
		resR = sp.ndimage.filters.gaussian_filter(r,sigma[2],output=sR)
		resG = sp.ndimage.filters.gaussian_filter(g,sigma[1],output=sG)
		resB = sp.ndimage.filters.gaussian_filter(b,sigma[0],output=sB)

		b = sB
		g = sG
		r = sR

	#I think the idea is to add sky shot noise *here*, after the sources have been convolved?
	if sigma_tuple[0] > 1.0e-8:
		print("Adding noise to b image: sigma = {:12.6f}".format(sigma_tuple[0]))
		b = b + sigma_tuple[0]*np.random.standard_normal(b.shape)

	if sigma_tuple[1] > 1.0e-8:
		print("Adding noise to g image: sigma = {:12.6f}".format(sigma_tuple[1]))
		g = g + sigma_tuple[1]*np.random.standard_normal(g.shape)

	if sigma_tuple[2] > 1.0e-8:
		print("Adding noise to r image: sigma = {:12.6f}".format(sigma_tuple[2]))
		r = r + sigma_tuple[2]*np.random.standard_normal(r.shape)

	b[sp.where(b <= 0.0)]=0.0 ; g[sp.where(g <= 0.0)]=0.0 ; r[sp.where(r <= 0.0)]=0.0
	
	I = (b+g+r)/3.0 + 1.0e-20
	
	
	minval = 0.0
	maxval = np.max(I)
	#print maxval
	
	#alph= 1.0  #fix alph to set the intensity of the faint features
	#Q=1.0  #1.0e-12 # 5.0  #e-12 #1.0e-10  #
	
	#10.0    #1.0e-10
	
	
	factor = little_f(I,minval,maxval,Q,alph)/I
	
	#R = r*factor
	#G = g*factor
	#B = b*factor
	R = little_f(r,minval,maxval,Q,alph)
	G = little_f(g,minval,maxval,Q,alph)
	B = little_f(b,minval,maxval,Q,alph)
	
	
	imarray = np.asarray([R,G,B])
	#print imarray.shape
	
	maxrgbval = np.amax(imarray, axis=0)
	#print maxrgbval.shape
	
	changeind = np.where(maxrgbval > 1.0)
	R[changeind] = R[changeind]/maxrgbval[changeind]
	G[changeind] = G[changeind]/maxrgbval[changeind]
	B[changeind] = B[changeind]/maxrgbval[changeind]
	
	#if maxrgb > 1.0:
	#	R=R/maxrgb
	#	G=G/maxrgb
	#	B=B/maxrgb
		
		
	ind = sp.where(I < 1.0e-10)
	R[ind]=0.0 ; G[ind]=0.0 ; B[ind]=0.0


	#imarray = np.asarray(np.transpose([sR,sG,sB]))
	imarray = np.asarray(np.transpose([R,G,B]))

	leny = float( len(R[0,:]))
	lenx = float( len(R[:,0]))
	inx = lenx/float(dpi)
	iny = leny/float(dpi)
	#print inx, iny
	
	return imarray



def make_quantity(im,filename,dpi=72,cmap='jet'):
	leny = float( len(im[0,:]))
	lenx = float( len(im[:,0]))
	inx = lenx/float(dpi)
	iny = leny/float(dpi)
	
	f1 = pyplot.figure(figsize=( inx, iny ), dpi=dpi, frameon=False)
	pyplot.subplots_adjust(bottom=0.0, top=1.0, left=0.0, right=1.0, hspace=0.0, wspace=0.0)
	#axi = pyplot.axes()
	#axi.set_xlim(0.5, 2.0)
	#axi.set_ylim(0.70, 1.40)
	#axi.locator_params(nbins=6,prune='both')
	#ax = axes([0,0,1,1], frameon=False)
	#ax.set_axis_off()

	axi=pyplot.axes([0.0,0.0,1.0,1.0], frameon=False)
	axi.set_axis_off()
	axi.imshow(np.transpose(im[:,:]),aspect='auto',cmap=cmap)
	#axi.get_xaxis().set_visible(False)
	#axi.get_yaxis().set_visible(False)
	#axi.set_frame_on(False)
	#if zlabel != -1:
	#	axi.annotate(str(zlabel),[0.7,0.9])
	#axi.set_xlabel('Observed Redshift')
	#axi.set_ylabel('$(U-V)_0$')
	
	f1.savefig(filename, dpi=dpi, format='pdf', pad_inches=0)

	pyplot.close(f1)


	
	return 0



def glob_broadband_images(bscale,gscale,rscale,filebase,alph,Q,inches=5.0,dpi=72,fwhm_pixels=0.0,sigma_tuple=[0.0,0.0,0.0],dirname='globbed_images',image_index=12,bind=0,gind=1,rind=3):
	if not os.path.exists(dirname):
		os.makedirs(dirname)

	bbfiles = np.sort(np.array(glob.glob('broadband*.fits')))#[0:10]

	count=0
	for fn in bbfiles:
		count = count + 1
		splitted = (fn.split('.fits'))[0]
		imagefile = os.path.join(dirname,filebase+splitted+'.pdf')
		hdulist = pyfits.open(fn)
		data = hdulist[image_index].data
		blue = bscale*data[bind,:,:]
		green = gscale*data[gind,:,:]
		red = rscale*data[rind,:,:]
		redshift = hdulist[1].header.get('REDSHIFT')
		res = make_general(bscale*blue,gscale*green,rscale*red,imagefile,alph,Q,inches,dpi,fwhm_pixels,sigma_tuple,zlabel=redshift)
	return 0

def make_gri(b,g,r,filename):

	
	b = b*1.5
	g = g*1.0
	r = r*1.0

	b[sp.where(b <= 0.0)]=0.0 ; g[sp.where(g <= 0.0)]=0.0 ; r[sp.where(r <= 0.0)]=0.0
	
	I = (b+g+r)/3.0 + 1.0e-20
	
	
	minval = 0.0
	maxval = np.max(I)
	#print maxval
	
	alph= 1.0  #fix alph to set the intensity of the faint features
	Q=1.0  #1.0e-12 # 5.0  #e-12 #1.0e-10  #
	
	#10.0    #1.0e-10
	
	
	factor = little_f(I,minval,maxval,Q,alph)/I
	
	R = r*factor
	G = g*factor
	B = b*factor
	
	
	imarray = np.asarray([R,G,B])
	#print imarray.shape
	
	maxrgbval = np.amax(imarray, axis=0)
	#print maxrgbval.shape
	
	changeind = np.where(maxrgbval > 1.0)
	R[changeind] = R[changeind]/maxrgbval[changeind]
	G[changeind] = G[changeind]/maxrgbval[changeind]
	B[changeind] = B[changeind]/maxrgbval[changeind]
	
	#if maxrgb > 1.0:
	#	R=R/maxrgb
	#	G=G/maxrgb
	#	B=B/maxrgb
		
		
	ind = sp.where(I < 1.0e-10)
	R[ind]=0.0 ; G[ind]=0.0 ; B[ind]=0.0
	imarray = np.asarray(np.transpose([R,G,B]))
	
	
	f1 = pyplot.figure(figsize=(5,5), dpi=20)

	#axi = pyplot.axes()
	#axi.set_xlim(0.5, 2.0)
	#axi.set_ylim(0.70, 1.40)
	#axi.locator_params(nbins=6,prune='both')
	axi=pyplot.axes([0.0,0.0,1.0,1.0])
	axi.imshow(imarray[50:150,50:150,:])
	#axi.set_xlabel('Observed Redshift')
	#axi.set_ylabel('$(U-V)_0$')
	
	f1.savefig(filename, format='pdf')

	pyplot.close(f1)


	
	return 0



def make_UVJ(b,g,r,filename):

	
	b = b*0.87
	g = g*0.5
	r = r*1.2

	b[sp.where(b <= 0.0)]=0.0 ; g[sp.where(g <= 0.0)]=0.0 ; r[sp.where(r <= 0.0)]=0.0
	
	I = (b+g+r)/3.0 + 1.0e-20
	
	
	minval = 0.0
	maxval = np.max(I)
	#print maxval
	
	alph=5.0e-1  #fix alph to set the intensity of the faint features
	Q= 5.0  #e-12 #1.0e-10  #
	
	#10.0    #1.0e-10
	
	
	factor = little_f(I,minval,maxval,Q,alph)/I
	
	R = r*factor
	G = g*factor
	B = b*factor
	
	
	imarray = np.asarray([R,G,B])
	#print imarray.shape
	
	maxrgbval = np.amax(imarray, axis=0)
	#print maxrgbval.shape
	
	changeind = np.where(maxrgbval > 1.0)
	R[changeind] = R[changeind]/maxrgbval[changeind]
	G[changeind] = G[changeind]/maxrgbval[changeind]
	B[changeind] = B[changeind]/maxrgbval[changeind]
	
	#if maxrgb > 1.0:
	#	R=R/maxrgb
	#	G=G/maxrgb
	#	B=B/maxrgb
		
		
	ind = sp.where(I < 1.0e-10)
	R[ind]=0.0 ; G[ind]=0.0 ; B[ind]=0.0
	imarray = np.asarray(np.transpose([R,G,B]))
	
	
	f1 = pyplot.figure(figsize=(5,5), dpi=50)

	#axi = pyplot.axes()
	#axi.set_xlim(0.5, 2.0)
	#axi.set_ylim(0.70, 1.40)
	#axi.locator_params(nbins=6,prune='both')
	axi=pyplot.axes([0.0,0.0,1.0,1.0])
	axi.imshow(imarray[75:325,75:325,:])
	#axi.set_xlabel('Observed Redshift')
	#axi.set_ylabel('$(U-V)_0$')
	
	f1.savefig(filename, format='eps')

	pyplot.close(f1)


	
	return 0


def make_IRAC(b,g,r,filename):

	
	b = b*0.07
	g = g*0.15
	#result = sp.ndimage.filters.gaussian_filter(r*1.0, 15.0, order=0, output=r)
	r = r*1.0
	
	b[sp.where(b <= 0.0)]=0.0 ; g[sp.where(g <= 0.0)]=0.0 ; r[sp.where(r <= 0.0)]=0.0
	
	I = (b+g+r)/3.0 + 1.0e-20
	
	
	minval = 0.0
	maxval = np.max(I)
	#print maxval
	
	alph= 50.0  #fix alph to set the intensity of the faint features
	Q= 4.0  # #1.0e-10  #
	
	#10.0    #1.0e-10
	
	
	factor = little_f(I,minval,maxval,Q,alph)/I
	
	R = sp.ndimage.filters.gaussian_filter(r*factor, 1.5, order=0)
	G = g*factor
	B = b*factor
	
	
	imarray = np.asarray([R,G,B])
	#print imarray.shape
	
	maxrgbval = np.amax(imarray, axis=0)
	#print maxrgbval.shape
	
	changeind = np.where(maxrgbval > 1.0)
	R[changeind] = R[changeind]/maxrgbval[changeind]
	G[changeind] = G[changeind]/maxrgbval[changeind]
	B[changeind] = B[changeind]/maxrgbval[changeind]
	
	#if maxrgb > 1.0:
	#	R=R/maxrgb
	#	G=G/maxrgb
	#	B=B/maxrgb
		
		
	ind = sp.where(I < 1.0e-10)
	R[ind]=0.0 ; G[ind]=0.0 ; B[ind]=0.0
	imarray = np.asarray(np.transpose([R,G,B]))
	
	
	f1 = pyplot.figure(figsize=(5,5), dpi=50)

	#axi = pyplot.axes()
	#axi.set_xlim(0.5, 2.0)
	#axi.set_ylim(0.70, 1.40)
	#axi.locator_params(nbins=6,prune='both')
	axi=pyplot.axes([0.0,0.0,1.0,1.0])
	axi.imshow(imarray[75:325,75:325,:])
	#axi.set_xlabel('Observed Redshift')
	#axi.set_ylabel('$(U-V)_0$')
	
	f1.savefig(filename, format='eps')

	pyplot.close(f1)


	
	return 0



