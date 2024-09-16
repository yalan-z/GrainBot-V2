import cv2
import matplotlib.pyplot
import matplotlib.pyplot as plt
import matplotlib.scale
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import pylab
import numpy as np
from PIL import Image
from torchvision.utils import save_image


img = cv2.imread('result/result.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,200,255, cv2.THRESH_BINARY  +cv2.THRESH_OTSU)
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
threshopen = cv2.erode(thresh, kernel1, iterations=2)
threshclose = cv2.dilate(threshopen, kernel2, iterations=2)
#cv2.imshow("thresh", threshclose) cv2.waitKey()
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(threshclose, connectivity=4)
area=stats[:,4]
fig = Image.open('result/result.jpg')
width=float(fig.width)
pixelarea=(2/width)**2 #different AFM scale
area=area*pixelarea
areal=area.tolist()

#plt.xlabel('Grain Area(um2)')
fig,ax=plt.subplots()
matplotlib.pyplot.hist(
areal, bins=40, edgecolor='k', range=(0,0.6))
ax.set_xscale('log')

matplotlib.pyplot.xlabel('Grain Area(um2)')
matplotlib.pyplot.ylabel('Counts')

matplotlib.pyplot.waitforbuttonpress()
output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
for i in range(1, num_labels):

    mask = labels == i
    output[:, :, 0][mask] = np.random.randint(0, 255)
    output[:, :, 1][mask] = np.random.randint(0, 255)
    output[:, :, 2][mask] = np.random.randint(0, 255)
cv2.imshow('oginal', output)
#cv2.waitKey()
cv2.imwrite('result\seg.png',output)