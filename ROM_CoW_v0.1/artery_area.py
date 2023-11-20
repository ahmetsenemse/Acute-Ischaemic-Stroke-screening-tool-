import numpy as np
from sklearn.linear_model import LinearRegression



def descending  (age):
    radius=(7.73+0.11*age-0.41+0.02*age+4.91*1.9)/2
    radius=radius/1000
    area=np.pi*np.square(radius)
    return area


def BCT(age):
    if age<=20:
        radius=9.9/2000
    elif age<=40:
        radius=10.7/2000
    elif age<=60:
        radius=11.4/2000
    else :
        radius=12.6/2000
    area=np.pi*np.square(radius)
    
    return area

def LS(age):
    if age<=20:
        radius=7.7/2000
    elif age<=40:
        radius=9.1/2000
    elif age<=60:
        radius=9.5/2000
    else :
        radius=9.8/2000
    area=np.pi*np.square(radius)
    
    return area

def CCA(age):
    if age<=20:
        radius=8/2000
    elif age<=40:
        radius=7.7/2000
    elif age<=60:
        radius=7.7/2000
    else :
        radius=8/2000
    area=np.pi*np.square(radius)
    
    return area

def VAL(age):
    r=-0.089
    if age<19:
        xmean=17.31
        ymean=3.04
        xstd=1.42
        ystd=0.61
    else: 
        xmean=21.92
        ymean=3.41
        xstd=1.31
        ystd=0.53
    
    B1=r*(ystd/xstd)
    B0=ymean-B1*xmean
    d=B0+age*B1
    radius=(d/2)/1000
    area=np.pi*np.square(radius)
    
    return area
        
def VAR(age):
    r=-0.076
    if age<19:
        xmean=17.31
        ymean=3.36
        xstd=1.42
        ystd=0.43
    else: 
        xmean=21.92
        ymean=2.88
        xstd=1.31
        ystd=0.53
    
    B1=r*(ystd/xstd)
    B0=ymean-B1*xmean
    d=B0+age*B1
    radius=(d/2)/1000
    area=np.pi*np.square(radius)
    
    return area

def ICAL(age):
    r=0.025
    if age<19:
        xmean=17.31
        ymean=4.21
        xstd=1.42
        ystd=0.31
    else: 
        xmean=21.92
        ymean=4.39
        xstd=1.31
        ystd=0.37
    
    B1=r*(ystd/xstd)
    B0=ymean-B1*xmean
    d=B0+age*B1
    radius=(d/2)/1000
    area=np.pi*np.square(radius)
    
    return area


def ICAR(age):
    r=0.031
    if age<19:
        xmean=17.31
        ymean=4.23
        xstd=1.42
        ystd=0.47
    else: 
        xmean=21.92
        ymean=4.24
        xstd=1.31
        ystd=0.42
    
    B1=r*(ystd/xstd)
    B0=ymean-B1*xmean
    d=B0+age*B1
    radius=(d/2)/1000
    area=np.pi*np.square(radius)
    
    return area


def ACA(age):
    r=-0.41
    
    xmean=45.07
    ymean=1.51
    xstd=11.63
    ystd=0.28
    
    B1=r*(ystd/xstd)
    B0=ymean-B1*xmean
    d=B0+age*B1
    radius=(d/2)/1000
    area=np.pi*np.square(radius)
    
    return area


def AComA(age):
    r=-0.15
    
    xmean=45.07
    ymean=1.05
    xstd=11.63
    ystd=0.37
    
    B1=r*(ystd/xstd)
    B0=ymean-B1*xmean
    d=B0+age*B1
    radius=(d/2)/1000
    area=np.pi*np.square(radius)
    
    return area


def PComA(age):
    r=-0.25
    
    xmean=45.07
    ymean=1.10
    xstd=11.63
    ystd=0.11
    
    B1=r*(ystd/xstd)
    B0=ymean-B1*xmean
    d=B0+age*B1
    radius=(d/2)/1000
    area=np.pi*np.square(radius)
    
    return area


def PCA(age):
    r=-0.45
    
    xmean=45.07
    ymean=1.85
    xstd=11.63
    ystd=0.23
    
    B1=r*(ystd/xstd)
    B0=ymean-B1*xmean
    d=B0+age*B1
    radius=(d/2)/1000
    area=np.pi*np.square(radius)
    
    return area

def MCA(age):
    r=-0.55
    
    xmean=45.07
    ymean=1.65
    xstd=11.63
    ystd=0.2
    
    B1=r*(ystd/xstd)
    B0=ymean-B1*xmean
    d=B0+age*B1
    radius=(d/2)/1000
    area=np.pi*np.square(radius)
    
    return area

def BA(age):
    r=0.67
    
    xmean=45.07
    ymean=3
    xstd=11.63
    ystd=0.11
    
    B1=r*(ystd/xstd)
    B0=ymean-B1*xmean
    d=B0+age*B1
    radius=(d/2)/1000
    area=np.pi*np.square(radius)
    
    return area
        
