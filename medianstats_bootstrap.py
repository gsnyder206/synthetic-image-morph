import math
import string
import sys
import struct
import numpy as np
import numpy.random as npr
import scipy.ndimage
import scipy.stats as ss
import scipy.signal
import scipy as sp
import gzip
import tarfile
import shutil
import numpy.ma as ma


def MAD(a, c=0.6745, axis=None):
    """
    Median Absolute Deviation along given axis of an array:

    median(abs(a - median(a))) / c

    c = 0.6745 is the constant to convert from MAD to std; it is used by
    default

    """

    a = ma.masked_where(a!=a, a)
    if a.ndim == 1:
        d = ma.median(a)
        m = ma.median(ma.fabs(a - d) / c)
    else:
        d = ma.median(a, axis=axis)
        # I don't want the array to change so I have to copy it?
        if axis > 0:
            aswp = ma.swapaxes(a,0,axis)
        else:
            aswp = a
        m = ma.median(ma.fabs(aswp.flatten() - d) / c, axis=0)

    return m



def bootstrap_median(sample, Nruns=1000):

    low = 0
    high = (sample.shape)[0] - 1
    medvals = np.ndarray(shape=(Nruns))*0.0

    for i in range(Nruns):
        rlist = npr.random_integers(low,high,high)
        current_sample = sample[rlist]
        current_median = np.median(current_sample)
        medvals[i] = current_median


    med_bestval = np.median(medvals)
    med_sigma = MAD(medvals)
    mad_sample = MAD(sample)
    return med_bestval, med_sigma, mad_sample
