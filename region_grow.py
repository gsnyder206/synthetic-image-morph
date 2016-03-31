import numpy as np
import scipy.ndimage
import scipy as sp

#direct translation of IDL code
def region_grow(array,roiPixels,ALL_NEIGHBORS=True,STDDEV_MULTIPLIER=None,THRESHOLD=None):
    arrDims = array.shape
    roiPixelData = array #.flatten()[roiPixels]
    nROIPixels = roiPixels.shape[0]

    thresh_lo = np.min(THRESHOLD)
    thresh_hi = np.max(THRESHOLD)

    #omitting stddev and mean thresholding option

    threshArray = np.where(np.logical_and(array >= thresh_lo,array <= thresh_hi),np.ones_like(array),np.zeros_like(array))
    labelArray = np.zeros_like(array)
    num_features = scipy.ndimage.measurements.label(threshArray,structure=[[1,1,1],[1,1,1],[1,1,1]],output=labelArray)

    #assuming for now that we're interested in the entire array (roiPixels not implemented)

    growROIPixels = np.where(labelArray > 0.0)[0]
    
    #num_features2 = scipy.ndimage.measurements.label(threshArray,structure=[[1,1,1],[1,1,1],[1,1,1]],output=labelArray)


    return growROIPixels, num_features, labelArray
