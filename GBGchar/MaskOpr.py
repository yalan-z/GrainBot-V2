import cv2
import numpy as np


def masking(label,lab_img):
    mask = lab_img == label
    masknpdata = np.zeros((256, 256), dtype=int)  # channel numbers
    masknpdata = masknpdata.astype(np.uint8)
    masknpdata[mask] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    masknpdata = cv2.dilate(masknpdata, kernel, iterations=2)

    return masknpdata


def maskafm(masknpdata, afm_img):
    cp_img = afm_img.copy()
    cp_mask = masknpdata.copy()
    mask = cp_mask == 255

    regionafm = np.zeros((256, 256), dtype=int)
    regionafm = regionafm.astype(np.uint8)
    regionafm[mask] = 1
    regionafm = regionafm * cp_img
    # cv2.imshow('region',regionafm)
    # cv2.waitKey()

    return regionafm


def find_min(afm_r):
    afm_r[afm_r == 0] = 255
    loc_min = afm_r.min()
    afm_r[afm_r == 255] = 0
    return (loc_min)


def exist_hier(afm_r, loc_min, s):
    th, afm_th = cv2.threshold(src=afm_r, \
                               thresh=loc_min + s, maxval=255, type=0)
    con, hier = cv2.findContours(afm_th, \
                                 cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return (not (hier[0][0][2] == -1))


def depression_cum(afm_r, loc_min, step):
    th, afm_th = cv2.threshold(src=afm_r, \
                               thresh=loc_min + step, maxval=255, type=0)
    con, hier = cv2.findContours(afm_th, \
                                 cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(con[-1])
    dep = 1
    while exist_hier(afm_th, th, s=step):
        th, afm_th = cv2.threshold(src=afm_r, \
                                   thresh=th + step, maxval=255, type=0)
        con, hier = cv2.findContours(afm_th, \
                                     cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        area += cv2.contourArea(con[-1])
        dep += 1
    return (area * dep * step)


def exist_nested(hier):
    for i in hier[0][..., 2]:
        if (i != -1):
            a = 1
            break
        else:
            a = 0
    return (a == 1)


def exist_ith_nested(ith_hier):
    if ith_hier[2] != -1:
        return (True)
    else:
        return (False)


def exist_samelev(depre, arg_index, hier, con):
    list_del = [arg_index]
    while hier[0][arg_index][0] != -1:
        arg_index = hier[0][arg_index][0]
        list_del.append(arg_index)
        depre += cv2.contourArea(con[arg_index])

    return list_del, depre


def remove_elements(list_orig, list_remove):
    for i in list_remove:
        list_orig.remove(i)
    return list_orig


def thresh_area_cal(bulge, depre, con, hier):
    # global d_area, b_area
    #print(bulge)
    #print(depre)
    if exist_nested(hier):
        index = list(range(0, hier.shape[1]))
        for i in index:
            if exist_ith_nested(hier[0][i]):
                nexti = hier[0][i][2]
                depre += cv2.contourArea(con[nexti])

                list_del, depre = exist_samelev(depre, nexti, hier, con)
                index = remove_elements(index, list_del)
            else:
                bulge += cv2.contourArea(con[i])
    else:
        index = list(range(0, hier.shape[1]))
        for i in index:
            bulge += cv2.contourArea(con[i])
    return bulge, depre


def BD_calculator(afm_region, step):
    th_min = afm_region.min() + step
    b_area = 0
    d_area = 0
    b_relat = 0

    afm_region[afm_region == 255] = 0
    th, afm_bi = cv2.threshold(src=afm_region, \
                               thresh=th_min, maxval=255, type=0)
    i = 0
    i_b = 0
    i_d = 0
    while th_min < afm_region.max():
        con_inf, hier_np = cv2.findContours( \
            afm_bi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        delta_b, delta_d = thresh_area_cal(b_area, d_area, con_inf, hier_np)
        i += 1
        th_min = th_min + step
        th, afm_bi = cv2.threshold(src=afm_region, \
                                   thresh=th_min, maxval=255, type=0)
        if d_area != delta_d:
            i_d += 1
        if delta_d != 0 and d_area == delta_d:
            i_b += 1
            b_relat += delta_b - b_area
            print('relative:' + str(b_relat))
        b_area = delta_b
        d_area = delta_d

    v_b = b_relat * i_b * step
    v_d = d_area * i_d * step

    return v_b, v_d
