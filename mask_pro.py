import numpy as np
import cv2

def mask_dilation(maskimg,label):
    mask = maskimg == label
    masknpdata = np.zeros((256, 256), dtype=int)  # channel numbers
    masknpdata = masknpdata.astype(np.uint8)
    masknpdata[mask] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    masknpdata = cv2.dilate(masknpdata, kernel, iterations=3)

    return masknpdata

def mask_afm(afmimg,masknpdata):
    cp_afmimg = afmimg.copy()
    submask = masknpdata.copy()
    mask = submask == 255
    regionafm = np.zeros((256, 256), dtype=int)
    regionafm = regionafm.astype(np.uint8)
    regionafm[mask] = 1
    regionafm = regionafm * cp_afmimg
    # cv2.imshow('region',regionafm)
    # cv2.waitKey()
    return regionafm