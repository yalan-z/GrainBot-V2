import cv2
import numpy as np
import math

class AFMimg:
    def __init__(self,afmrgb,maskimg,lmin,lmax,width):
        self.maskimg = maskimg

        self.afm = cv2.cvtColor(afmrgb, cv2.COLOR_BGR2GRAY)
        depthimg = self.afm.copy()
        self.lmin = lmin
        self.lmax = lmax
        self.realdimg = ((self.lmax - self.lmin) / (255 - 0)) * (depthimg - 0) + self.lmin

        self.pixelsize = width / 256 #in nanometer
    def MorphOpr(self):
        img = cv2.cvtColor(self.maskimg, cv2.COLOR_BGR2GRAY)
        ret, imthresh = cv2.threshold(img, 253, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (4,4))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        self.maskimg = cv2.erode(imthresh, kernel1, iterations=2)
        #self.maskimg = cv2.dilate(erosion, kernel2, iterations=3)
    def Labeling(self):
        self.numx, self.labelimg, stat, centroid = \
            cv2.connectedComponentsWithStats(self.maskimg, connectivity=4)
    def Preseg(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2))
        self.hardseg=cv2.erode(self.maskimg, kernel, iterations=3)
        
        


