import cv2
import numpy as np
from AFMOpr import AFMimg
import MaskOpr
import BoundaryOpr as BO


class Grains():

    def __init__(self, label):

        self.label = label


    def grainmask(self, labelimg,afmimg):
        self.mask = MaskOpr.masking(self.label, labelimg)
        self.afm = MaskOpr.maskafm(self.mask, afmimg)

    def maskero(self,afmimg):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.maske = cv2.erode(self.maskd, kernel, iterations=4)
        self.afme = MaskOpr.maskafm(self.maske, afmimg)
        self.afme[self.afme == 0] = 255
    def maskregain(self,afmimg):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.maskre = cv2.erode(self.maskd, kernel, iterations=3)
        self.afmre = MaskOpr.maskafm(self.maskre, afmimg)
    def maskdil(self,afmimg):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.maskd = cv2.dilate(self.mask, kernel, iterations=3)
        self.afmd = MaskOpr.maskafm(self.maskd, afmimg)
    def areacal(self):
        area=np.sum(self.maskre==255)
        self.area=area
    def voidcal(self, step):
        th_min = self.afme.min() + step
        b_area = 0
        d_area = 0
        b_relat = 0

        self.afme[self.afme == 255] = 0
        th, afm_bi = cv2.threshold(src=self.afme, \
                                   thresh=th_min, maxval=255, type=0)
        i = 0
        i_b = 0
        i_d = 0
        while th_min < self.afme.max():
            con_inf, hier_np = cv2.findContours( \
                afm_bi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            delta_b, delta_d = MaskOpr.thresh_area_cal(b_area, d_area, con_inf, hier_np)
            i += 1
            th_min = th_min + step
            th, afm_bi = cv2.threshold(src=self.afme, \
                                       thresh=th_min, maxval=255, type=0)
            if d_area != delta_d:
                i_d += 1
            if delta_d != 0 and d_area == delta_d:
                i_b += 1
                b_relat += delta_b - b_area
                # print('relative:' + str(b_relat))
            b_area = delta_b
            d_area = delta_d

        self.rel_vb = b_relat * step
        self.vd = d_area * step
        self.vb = b_area * step

    def anglecal(self,realdimg,pixelsize):
        maskcnt, maskhei = cv2.findContours( \
            self.maskd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        grdlist, bdrlist = BO.gradientonxy(maskcnt)
        tailbdr = BO.tailorbdrlist(bdrlist, maskcnt)
        

        if tailbdr != []:
            result_tan,self.grdplot,delta_z=\
        BO.boundaryfit(tailbdr,grdlist,maskcnt,realdimg,pixelsize)
            #print(delta_z)
            self.clean_tan=BO.anglecleaning(result_tan)
            
            if self.clean_tan!= []:
                self.angles,self.tan = BO.convertangle(self.clean_tan)
                
                self.aveangle=sum(self.angles)/len(self.angles)
                #print(self.angles)
            
                #self.clean_tan=BO.anglecleaning(result_tan)
                #BO.check_grd_selection(realdimg, grdplot)
            else:
                self.angles=[0]
        else:
            self.angles=[0]
           


           #self.resultlist=result_tan
    def angleselect(self):
        self.aveangle=sum(self.angles)/len(self.angles)
        
