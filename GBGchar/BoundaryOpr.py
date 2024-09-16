import cv2
import numpy as np
import math
from numpy import uint8

'''
image information
pixel size: 2 um/256 pixels
depth scale: 255:30=9.7nm:-10.4nm
'''

def mask_dilation(label):
    mask = maskimg == label
    masknpdata = np.zeros((256, 256), dtype=int)  # channel numbers
    masknpdata = masknpdata.astype(np.uint8)
    masknpdata[mask] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    masknpdata = cv2.dilate(masknpdata, kernel, iterations=3)

    return masknpdata


def viewmask(label):  ##change variables
    img = cv2.addWeighted(origimg, 1, masknpdata[:, :], 0.3, 0)
    cv2.imshow('maskedAFM', img)
    cv2.waitKey()


def maskafm(masknpdata):
    cporigimg = origimg.copy()
    submask = masknpdata.copy()
    mask = submask == 255
    regionafm = np.zeros((256, 256), dtype=int)
    regionafm = regionafm.astype(np.uint8)
    regionafm[mask] = 1
    regionafm = regionafm * cporigimg
    # cv2.imshow('region',regionafm)
    # cv2.waitKey()
    return regionafm


def gradientonxy(coord_list):
    gradlist = []
    linelist = [[]]
    crd = []
    j = 0
    for i in range(0, len(coord_list[0])-3, 3):
        y0 = coord_list[0][i][0][1]
        x0 = coord_list[0][i][0][0]
        yt = coord_list[0][i + 3][0][1]
        xt = coord_list[0][i + 3][0][0]
        if (x0 - xt) == 0:
            gradlist.append(999)
            # crd[0]
            # crdlist[i].append()
        if (x0 - xt) != 0:
            grad = (y0 - yt) / (x0 - xt)
            gradlist.append(grad)
            # print(gradlist)
        if i > 0:
            if gradlist[int(i / 3)] * gradlist[int(i / 3) - 1] > 0:
                linelist[j].append(i)

            if gradlist[int(i / 3)] * gradlist[int(i / 3) - 1] < 0:
                linelist.append([])
                j += 1
                linelist[j].append(i)
    for d in range(0, len(linelist) - 1):
        if len(linelist[d]) < 2:
            linelist[d] = []
    for n in range(0,len(linelist)):
        while [] in linelist[n]:
            linelist[n].remove([])
    while [] in linelist:
        linelist.remove([])


    return gradlist, linelist


def morethan2(x):
    return len(x) > 2


def removempty(x):
    return [] in x


def tailorbdrlist(bdrlist, coor_list):
    for i in range(0, len(bdrlist)):
        for j in bdrlist[i]:
            if coor_list[0][j][0][0] > 245 \
                    or coor_list[0][j][0][0] < 10:
                bdrlist[i][bdrlist[i].index(j)] = []

            elif coor_list[0][j][0][1] > 245 \
                    or coor_list[0][j][0][1] < 10:
                bdrlist[i][bdrlist[i].index(j)]=[]
    for i in range(0, len(bdrlist)):
        while [] in bdrlist[i]:
            bdrlist[i].remove([])
    for i in range(0, len(bdrlist)):
        if len(bdrlist[i])<4:
            bdrlist[i]=[]
    while [] in bdrlist:
        bdrlist.remove([])

    return bdrlist




def boundaryfit(boundlist, grdlist, coord_list,realdepth,pixelsize):  # bdrlist,testlist,coordinates
    gradlist = []
    coor_z_list = [[]]
    check_z_list = [[]]
    grdarray = np.zeros((256, 256), dtype=int)
    boundlist = list(filter(morethan2, boundlist))

    for i in range(0, len(boundlist)):
        coor_z_list.append([])
        check_z_list.append([])
        delx2 = boundlist[i][-1]
        delx1 = boundlist[i][0]
        if coord_list[0][delx2][0][0] - coord_list[0][delx1][0][0] > 0:
            cofa = 1
        else:
            cofa = -1
        sum = 0
        n = 0
        for j in range(0, len(boundlist[i])):
            if grdlist[int(boundlist[i][j] / 3)] != 999:
                sum += grdlist[int(boundlist[i][j] / 3)]
                n += 1
                # print(i,j,sum,n)
        avggrad = sum / n
        # print(avggrad)
        gbgrd = -1 / avggrad
        # print(i,gbgrd)
        gradlist.append(gbgrd)
        grdslice = grdlist[int(min(boundlist[i]) / 3):int(max(boundlist[i]) / 3)]
        if grdslice.count(999) > 5:
            for j in boundlist[i]:
                jindex = boundlist[i].index(j)
                coor_z_list[i].append([])
                check_z_list[i].append([])

                y1 = coord_list[0][j][0][1]
                x1 = coord_list[0][j][0][0]
                x2 = x1
                y2 = y1
                for n in range(0, 6):
                    y3 = y2
                    x3 = x2 - 1
                    # print(j, x3)
                    z1 = realdepth[y2, x2]
                    z2 = realdepth[y3, x3]
                    d = math.sqrt((y2 - y3) ** 2 + (x2 - x3) ** 2) * pixelsize
                    grd_element = (z2 - z1) / d
                    # print(grd_element)

                    coor_z_list[i][jindex].append(grd_element)
                    check_z_list[i][jindex].append(z2 - z1)
                    y2 = y3
                    x2 = x3
                while [] in coor_z_list[i]:
                    coor_z_list[i].remove([])




        else:

            for j in boundlist[i]:
                jindex = boundlist[i].index(j)
                coor_z_list[i].append([])
                check_z_list[i].append([])

                y1 = coord_list[0][j][0][1]
                x1 = coord_list[0][j][0][0]
                x2 = x1
                y2 = y1

                for n in range(0, 6):
                    y3 = y2 - cofa
                    if y3 > 255 or y3 < 0:
                        break

                    x3 = int((y3 - y2) / gbgrd + x2)
                    if x3 > 255 or x3 < 0:
                        break
                    grdarray[y3][x3] = 255
                    # print(j, x3)
                    z1 = realdepth[y2, x2]
                    z2 = realdepth[y3, x3]
                    d = math.sqrt((y2 - y3) ** 2 + (x2 - x3) ** 2) * pixelsize
                    grd_element = (z2 - z1) / d
                    # print(grd_element)

                    coor_z_list[i][jindex].append(grd_element)
                    check_z_list[i][jindex].append(z2 - z1)
                    y2 = y3
                    x2 = x3
                while [] in coor_z_list[i]:
                    coor_z_list[i].remove([])

    while [] in coor_z_list:
        coor_z_list.remove([])
    grdarray = grdarray.astype(np.uint8)
    #print(coor_z_list)
    return coor_z_list, grdarray, check_z_list

def anglecleaning(coor_z_list):
    i=0
    
    while i < len(coor_z_list):
        j=0
        while j < len(coor_z_list[i]):
            del coor_z_list[i][j][0]
            x=0
            while x < len(coor_z_list[i][j]):
                if coor_z_list[i][j][x]<=0:
                    del coor_z_list[i][j][x]
                else:
                    x+=1
            j+=1
        while [] in coor_z_list[i]:
            coor_z_list[i].remove([])
        i+=1
    while [] in coor_z_list:
        coor_z_list.remove([])
    return coor_z_list
            
'''           
            
    for i in range(0, len(coor_z_list)):
        for j in range(0, len(coor_z_list[i])):
            del coor_z_list[i][j][0]
            for x in coor_z_list[i][j]:
                if x<=0:
                    coor_z_list[i][j].remove(x)
                else:
                    continue
        while [] in coor_z_list[i]:
            coor_z_list[i].remove([])
              
    return coor_z_list    
'''

def check_grd_selection(img, grdarray):
    cpimg = img.copy()
    while 0 in grdarray:
        grdarray[grdarray == 0] = 1
    while 255 in grdarray:
        grdarray[grdarray == 255] = 0
    grdarray = grdarray * cpimg
    cv2.imshow("visual", grdarray)

    #cv2.imwrite(os.path.join(path, 'visualboundary.jpg'), grdarray)
    cv2.waitKey()


def check_brdlist(bdrlist, img):
    cpimg = img.copy()
    for i in bdrlist:
        cpimg[maskcnt[0][i[0]][0]] == 255
        # print(maskcnt[0][i[0]][0])
        cpimg[maskcnt[0][i[-1]][0]] == 255
        # print(maskcnt[0][i[-1]][0])
    fig = px.imshow(cpimg)
    fig.show()
# check_brdlist(bdrlist, origimg)

def convertangle(resultlist):
    anglelist = []

    for i in range(0, len(resultlist)):
        #anglelist.append([])

        for j in range(0, len(resultlist[i])):
            anglelist.append(max(resultlist[i][j]))
            # anglelist[i].append(min(resultlist[i][j]))
    degree_aglst=[0]
    for r in range(0,len(anglelist)):
        degree_aglst[r] = math.degrees(math.atan(anglelist[r]))
        #print(math.atan(anglelist[r]))
        degree_aglst.append(0)
    del degree_aglst[-1]

      
    return degree_aglst,anglelist
'''
all_list = []
all_delta_z = []
a = 0
for l in range(1, maskimg.max() + 1):
    all_list.append([])
    all_delta_z.append([])
    masknpdata = mask_dilation(l)
    # viewmask(l)
    afmregions = maskafm(masknpdata)
    maskcnt, maskhei = cv2.findContours(masknpdata, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # number of label
    cporigimg = origimg.copy()

    grdlist, bdrlist = gradientonxy(maskcnt)
    tailbdr=tailorbdrlist(bdrlist, maskcnt)

    if tailbdr!=[]:
        result_tan, grdplot, delta_z = boundaryfit(tailbdr, grdlist, maskcnt)
        #check_grd_selection(origimg, grdplot)


        g1 = Angle.GBGangle(result_tan)
        ang_list = g1.GBGcal()
        all_list[l - 1].append(ang_list)
        all_delta_z[l - 1].append(delta_z)


plot_all_list = all_list.copy()
tailed_all_list, tailed_oned = plots.tailer(plot_all_list)
plots.histgrm(tailed_oned, 20)
'''
'''
cv2.drawContours(cporigimg, maskcnt, -1, (0, 0, 255), 1)

cv2.imshow('show', cporigimg)
#cv2.imwrite(os.path.join(path, 'AFMboundary.jpg'), cporigimg)
cv2.waitKey()
check_grd_selection(origimg,grdplot)
'''
'''
def visual_angle(origimg,angle_list):

    fig = px.imshow(origimg, binary_string=True)
    fig.update_traces(hoverinfo='skip')  # hover is only for label info
    for index in range(1, label.max()):
        label_i = props[index].label
        contour = measure.find_contours(label == label_i, 0.5)[0]
        y, x = contour.T
        hoverinfo = ''

        hoverinfo += f'<b>{properties}: {getattr(props[index], properties)}</b><br>'
        fig.add_trace(go.Scatter(
            x=x, y=y, name=label_i,
            mode='lines', fill='toself', showlegend=False,
            hovertemplate=hoverinfo, hoveron='points+fills'))

    plotly.io.show(fig)
    return
'''

'''
with open(txtfile, "w") as f:
    for line in tailed_oned:
        f.write(str(line) + "\n")
'''

