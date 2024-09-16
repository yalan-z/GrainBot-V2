import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from scipy.optimize import curve_fit
import numpy as np

class StatisticAng:
    def __init__(self,cpath):
        self.pf=pd.read_csv(cpath)
        self.data=self.pf['angle'].values.tolist()
        self.mu = np.mean(self.data)
        self.sigma = np.std(self.data)
    def histplot(self,xmin,xmax):


        font={"family":"Arial","style":"normal","weight":"normal","color":"black","size":18}
        plt.hist(self.data, bins=40, range=(xmin,xmax),density=True, histtype='barstacked', rwidth=0.7,
                                    edgecolor='royalblue', color='lightsteelblue')

        x = np.arange(0, xmax, 0.5)
        y = norm.pdf(x, self.mu,self.sigma)
        plt.plot(x, y, color='darkolivegreen', linestyle='--')
        plt.title("Distribution of GBG Angle", fontdict=font, pad=18)
        plt.xlabel('GBG angle (°)', fontdict=font)
        plt.ylabel('Probability density', fontdict=font)
        plt.tick_params('both', labelsize=16)
        plt.ticklabel_format(style='sci', scilimits=(-1, 2), axis='x')
    def mean(self):
        print('Mean value of the GBG angle: ' + '%.2f' % self.mu)

    def variance(self):
        print('Variance of the GBG angle: ' + '%.2f' % self.sigma)

class StatisticCCA:
    def __init__(self,cpath):
        self.pf=pd.read_csv(cpath)
        self.cca=self.pf['cca'].values.tolist()
        self.mu = np.mean(self.cca)
        self.sigma = np.std(self.cca)
    def histplot(self,numb=None,ran=None):
        if ran is None:
            ran=(20,max(self.cca))
        if numb is None:
            numb=20
        font={"family":"Arial","style":"normal","weight":"normal","color":"black","size":18}
        plt.hist(self.cca, bins=numb,range=ran,histtype='barstacked', rwidth=0.7,
                                    edgecolor='royalblue', color='lightsteelblue')


        plt.title("Distribution of Grain Concavity", fontdict=font, pad=18)
        plt.xlabel('Concavity volume (nm^3)', fontdict=font)
        plt.ylabel('Number', fontdict=font)
        plt.ticklabel_format(style='sci',scilimits=(-1,2),axis='x')
        plt.tick_params('both', labelsize=16)



    def mean(self):
        print('Mean value of concavity volume: ' + '%.2f' % self.mu)
    def variance(self):
        print('Variance of concavity volume: ' + '%.2f' % self.sigma)
    def ratio(self):
        self.ratio = 1 - self.cca.count(0) / (len(self.cca) - 1)
        print('Concavity ratio: ' + '%.4f' % self.ratio)

class StatisticCVE:
    def __init__(self,cpath):
        self.pf = pd.read_csv(cpath)
        self.cve = self.pf['cve'].values.tolist()
        self.mu = np.mean(self.cve)
        self.sigma = np.std(self.cve)
    def histplot(self,numb=None,ran=None):
        if ran is None:
            ran=(20,max(self.cve))
        if numb is None:
            numb=20

        font={"family":"Arial","style":"normal","weight":"normal","color":"black","size":18}
        plt.hist(self.cve, bins=numb,range=ran, histtype='barstacked', rwidth=0.7,
                                    edgecolor='royalblue', color='lightsteelblue')


        plt.title("Distribution of Grain Convexity", fontdict=font, pad=18)
        plt.xlabel('Concavity volume (nm^3)', fontdict=font)
        plt.ylabel('Number', fontdict=font)
        plt.ticklabel_format(style='sci', scilimits=(-1, 2), axis='both')
        plt.tick_params('both', labelsize=16)

    def mean(self):
        print('Mean value of concavity volume: ' + '%.2f' % self.mu)
    def variance(self):
        print('Variance of concavity volume: ' + '%.2f' % self.sigma)

class StatisticGSA:
    def __init__(self,cpath):
        self.pf = pd.read_csv(cpath)
        gsa = self.pf['gsa'].values.tolist()
        gsal=[]
        for i in gsa:
            gsal.append(i/1e6)
        self.gsa=gsal
    def histplot(self,numb=None,ran=None):
        if ran is None:
            ran=(0,max(self.gsa))
        if numb is None:
            numb=20

        font={"family":"Arial","style":"normal","weight":"normal","color":"black","size":18}
        self.n,binx,patches=plt.hist(self.gsa, bins=numb,range=ran,density=True, histtype='barstacked', rwidth=0.7,
                                    edgecolor='royalblue', color='lightsteelblue')
        self.bins=binx[1:]

        plt.title("Distribution of Grain Surface Area", fontdict=font, pad=18)
        plt.xlabel('Grain Surface Area (um^2)', fontdict=font)
        plt.ylabel('Number', fontdict=font)
        plt.ticklabel_format(style='sci', scilimits=(-1, 2), axis='both')
        plt.tick_params('both', labelsize=16)

        def func(x,a,k,c):
            return a*np.exp(x**k)+c

        para,cov=curve_fit(func,self.bins,self.n,p0=[0,-0.01,1])
        self.k=para[1]
        self.s=max(self.gsa)
        x1 = np.arange(0, self.s, 0.01)
        y = func(x1,para[0],para[1],para[2])
        plt.plot(x1, y, color='darkolivegreen', linestyle='--')
        print(self.k,self.s)




