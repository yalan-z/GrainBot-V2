import cv2
from skimage import data, measure, morphology
import numpy as np
import matplotlib.pyplot as plt
import random
import plotly
import plotly.express as px
import plotly.graph_objects as go
import time
import pdb
import math


impath=''
imname=''
rgb=''
image=cv2.imread(impath+'\\'+imname)
def preprocess(img,k1,k2,numi):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, imthresh=cv2.threshold(img, 253, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, k1)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k2)
    erosion = cv2.erode(imthresh, kernel1, iterations=numi)
    imageopen = cv2.dilate(erosion, kernel2, iterations=3)
    return imageopen
def labeling(img):
  numx,labels,stat,centroid=\
  cv2.connectedComponentsWithStats(img, connectivity=4)
  return labels
def openlarge(bimage,k5,k6,numi):

    kernel5 = cv2.getStructuringElement(cv2.MORPH_CROSS, k5)
    kernel6 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k6)
    erosion = cv2.erode(bimage, kernel5, iterations=numi)
    imageopen = cv2.dilate(erosion, kernel6, iterations=1)
    return imageopen
def largeerosion(bimage,k7,k8,numi):

    kernel7 = cv2.getStructuringElement(cv2.MORPH_RECT, k7)
    kernel8 = cv2.getStructuringElement(cv2.MORPH_CROSS, k8)
    erosion = cv2.erode(bimage, kernel7, iterations=numi)
    imageopen = cv2.dilate(erosion, kernel8, iterations=1)
    return imageopen
def close(bimage,k3,k4,numi):
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k3)
    kernel4 = cv2.getStructuringElement(cv2.MORPH_CROSS, k4)
    dilation = cv2.dilate(bimage, kernel3, iterations=numi)
    imageclose = cv2.erode(dilation, kernel4, iterations=1)
    return imageclose

def visualabels(image,num_labels,labels,savename):
    labelplot = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    for i in range(1, num_labels):
        mask = labels == i
        j = random.randrange(0, len(rgb) - 1)
        labelplot[:, :, 0][mask] = rgb[j][2]
        labelplot[:, :, 1][mask] = rgb[j][1]
        labelplot[:, :, 2][mask] = rgb[j][0]
    cv2.imshow('label', labelplot)
    cv2.waitKey()
    cv2.imwrite(impath + '\\' + savename +'.jpg', labelplot)
    cv2.imwrite(impath + '\\' + savename + '.png', labels)
    return
def visuallization(img,label,props,properties):
    fig = px.imshow(img, binary_string=True)
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
def distance(centerp,regionp):
    dpoint=regionp-centerp
    d=np.sqrt(np.sum(np.square(dpoint)))
    return d
def labelcontour(label,num_labels):
    arraycon=np.empty([num_labels-1,1])
    labelarray = np.zeros((image.shape[0], image.shape[1]),np.float_)
    for i in range(1, num_labels):
        mask = label == i
        labelarray[:, :][mask] = 255
        contours=measure.find_contours(labelarray)
        arraycon[i-1]=contours
    return arraycon

'''
#lightclose=close(imthresh,(1,1),(1,1),1)
lightopen=open(imthresh,(4,4),(2,2),2)
afteropen=openlarge(imthresh,(5,5),(1,1),2)

numex,labelextract,statex,centroidfalt=cv2.connectedComponentsWithStats(lightopen, connectivity=4)
numl,labelatlarge,statero,centroids=cv2.connectedComponentsWithStats(afteropen, connectivity=4)

visualabels(image,numex,labelextract,'lightopen')
visualabels(image,numl,labelatlarge,'largeopen')
propsero = measure.regionprops(labelatlarge)
propex=measure.regionprops(labelextract)
properties = 'centroid'
#visuallization(image,labelatlarge,propsero,properties)
listcen=[]
n=1
labelmax=labelextract.max()
for j in range(1,labelmax+1):
    listcen = []
    labelarray = np.zeros((image.shape[0], image.shape[1]), np.float_)
    mask = labelextract == j
    labelarray[:, :][mask] = 255
    contours = measure.find_contours(labelarray)[0]

    for i in range(0,labelatlarge.max()-1):
        points=np.array([[propsero[i].centroid[0],propsero[i].centroid[1]]])
        if measure.points_in_poly(points,contours):
            listcen.append(propsero[i].centroid)
    #print(listcen)
    if len(listcen)==0:
        print('listcen=0 at j='+str(j))
    if len(listcen)==1:
        continue
    if len(listcen)==2:
        for i2 in range(0,propex[j-1].area):
            d1 = distance(np.array(listcen[0]), propex[j-1].coords[i2])
            d2 = distance(np.array(listcen[1]), propex[j-1].coords[i2])
            if d1 > d2:
                labelextract[propex[j-1].coords[i2][0],propex[j-1].coords[i2][1]] = n + labelmax
        n += 1
    if len(listcen)==3:
        m=n+1
        for i2 in range(0, propex[j-1].area):
            d1 = distance(np.array(listcen[0]), propex[j-1].coords[i2])
            d2 = distance(np.array(listcen[1]), propex[j-1].coords[i2])
            d3 = distance(np.array(listcen[2]), propex[j - 1].coords[i2])
            if d1>d2 and d2>d3:
                continue
            if d1>d2 and d3>d2:
                labelextract[propex[j-1].coords[i2][0], propex[j-1].coords[i2][1]] = n + labelmax
            if d1<d2 and d3>d1:
                labelextract[propex[j - 1].coords[i2][0], propex[j - 1].coords[i2][1]] = m + labelmax
            if d1<d2 and d3<d1:
                continue
        n=m+1


visualabels(image,numex+n-1,labelextract,'final')
'''













