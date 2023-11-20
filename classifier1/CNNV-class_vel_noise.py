import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import numpy as np
import tensorflow as tf
from tensorflow import keras
import datetime
from sklearn.metrics import f1_score
import artery_area as ata
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
import random

DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)

N_mu=500
numbermodes=1
n_features = 3

train_size=8000

a = np.ones(1)*0.2802 #
b =  np.ones(1)*(-0.5053)*1000 #m-1
c =  np.ones(1)*0.1324 #
d =  np.ones(1)*(-0.01114)*1000 #m-1
E=1.6*1e6

ACALvessels=np.loadtxt('..\Snaps\CoW_0_ACAL.csv',delimiter=',')
MCALvessels=np.loadtxt('..\Snaps\CoW_0_MCAL.csv',delimiter=',')
PCALvessels=np.loadtxt('..\Snaps\CoW_0_PCAL.csv',delimiter=',')
ACARvessels=np.loadtxt('..\Snaps\CoW_0_ACAR.csv',delimiter=',')
MCARvessels=np.loadtxt('..\Snaps\CoW_0_MCAR.csv',delimiter=',')
PCARvessels=np.loadtxt('..\Snaps\CoW_0_PCAR.csv',delimiter=',')

#### input data #####            
for j in range(0,N_mu):
    for i in range(0,130):
        name='..\Snaps\Age'+str(j)+'.npy'
        if os.path.isfile(name):
            age1=np.load(name)
        else:
            age1=np.linspace(30, 80, num=10)[(j%10)]
            
        A0=ata.CCA(age1)
        r=np.sqrt(A0/np.pi)
        h=r*(a*np.exp(b*r)+c*np.exp(d*r))
        beta=(4/3)*((np.sqrt(np.pi)*E*h)/(A0))
        
        if i==0:
            path="..\Snaps\CoW_"+str(j)+"_"+str(i)+"0.csv"
            if os.path.isfile(path):
                name1="..\Snaps\CoW_"+str(j)+"_00.csv"
                vessel1 = np.loadtxt(name1,delimiter=',')
                name1="..\Snaps\CoW_"+str(j)+"_01.csv"
                vessel2 = np.loadtxt(name1,delimiter=',')
                
                if j==20 or j ==21 or j ==22 or j==23:
                    pressure1=np.square(np.delete(vessel1, [0,1,2,3,5], 1)/beta+np.sqrt(A0))
                    pressure2=np.square(np.delete(vessel2, [0,1,2,3,5], 1)/beta+np.sqrt(A0))
                    
                    flowrate1=np.delete(vessel1, [0,1,3,4,5], 1)
                    flowrate2=np.delete(vessel2, [0,1,3,4,5], 1)
                else:
                    pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                    pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
                    
                    flowrate1=np.delete(vessel1, [0,2,3], 1)
                    flowrate2=np.delete(vessel2, [0,2,3], 1)
                
                
    
                age=np.tile(age1,(len(flowrate1),1))
            
                
                target=np.array([1,0,0,0,0,0,0,0,0])
        
                    
                if i==0 and j==0:
                    feature1=pressure1
                    feature2=pressure2
                    feature3=flowrate1
                    feature4=flowrate2
                    feature5=age
                    target_dataset=target
                else :
                    feature1=tf.concat([feature1,pressure1],axis=0)
                    feature2=tf.concat([feature2,pressure2],axis=0)
                    feature3=tf.concat([feature3,flowrate1],axis=0)
                    feature4=tf.concat([feature4,flowrate2],axis=0)
                    feature5=tf.concat([feature5,age],axis=0)
                    target_dataset=tf.concat([target_dataset,target],axis=0)
                    
            path="..\Snaps\CoW_"+str(j)+"_ACOR"+str(i)+"0.csv"
            if os.path.isfile(path):
                name1="..\Snaps\CoW_"+str(j)+"_ACOR00.csv"
                vessel1 = np.loadtxt(name1,delimiter=',')
                name1="..\Snaps\CoW_"+str(j)+"_ACOR01.csv"
                vessel2 = np.loadtxt(name1,delimiter=',')
                
                pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
                
                flowrate1=np.delete(vessel1, [0,2,3], 1)
                flowrate2=np.delete(vessel2, [0,2,3], 1)
                
                
                age=np.tile(age1,(len(flowrate1),1))
            
                
                target=np.array([0,1,0,0,0,0,0,0,0])
        
                    
                feature1=tf.concat([feature1,pressure1],axis=0)
                feature2=tf.concat([feature2,pressure2],axis=0)
                feature3=tf.concat([feature3,flowrate1],axis=0)
                feature4=tf.concat([feature4,flowrate2],axis=0)
                feature5=tf.concat([feature5,age],axis=0)
                target_dataset=tf.concat([target_dataset,target],axis=0)
                    
            path="..\Snaps\CoW_"+str(j)+"_ACOL"+str(i)+"0.csv"
            if os.path.isfile(path):
                name1="..\Snaps\CoW_"+str(j)+"_ACOL00.csv"
                vessel1 = np.loadtxt(name1,delimiter=',')
                name1="..\Snaps\CoW_"+str(j)+"_ACOL01.csv"
                vessel2 = np.loadtxt(name1,delimiter=',')
                
                pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
                
                flowrate1=np.delete(vessel1, [0,2,3], 1)
                flowrate2=np.delete(vessel2, [0,2,3], 1)
                
                
                age=np.tile(age1,(len(flowrate1),1))
            
                
                target=np.array([0,0,1,0,0,0,0,0,0])
        
                    
                feature1=tf.concat([feature1,pressure1],axis=0)
                feature2=tf.concat([feature2,pressure2],axis=0)
                feature3=tf.concat([feature3,flowrate1],axis=0)
                feature4=tf.concat([feature4,flowrate2],axis=0)
                feature5=tf.concat([feature5,age],axis=0)
                target_dataset=tf.concat([target_dataset,target],axis=0)
                
            

            path="..\Snaps\CoW_"+str(j)+"_ICAL"+str(i)+"0.csv"
            if os.path.isfile(path):
                name1="..\Snaps\CoW_"+str(j)+"_ICAL00.csv"
                vessel1 = np.loadtxt(name1,delimiter=',')
                name1="..\Snaps\CoW_"+str(j)+"_ICAL01.csv"
                vessel2 = np.loadtxt(name1,delimiter=',')
                
                pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
                
                flowrate1=np.delete(vessel1, [0,2,3], 1)
                flowrate2=np.delete(vessel2, [0,2,3], 1)
                
                
                age=np.tile(age1,(len(flowrate1),1))
            
                
                target=np.array([0,0,1,0,0,0,0,0,0])
        
                    
                feature1=tf.concat([feature1,pressure1],axis=0)
                feature2=tf.concat([feature2,pressure2],axis=0)
                feature3=tf.concat([feature3,flowrate1],axis=0)
                feature4=tf.concat([feature4,flowrate2],axis=0)
                feature5=tf.concat([feature5,age],axis=0)
                target_dataset=tf.concat([target_dataset,target],axis=0)
                
                
            path="..\Snaps\CoW_"+str(j)+"_ICAL2"+str(i)+"0.csv"
            if os.path.isfile(path):
                name1="..\Snaps\CoW_"+str(j)+"_ICAL200.csv"
                vessel1 = np.loadtxt(name1,delimiter=',')
                name1="..\Snaps\CoW_"+str(j)+"_ICAL201.csv"
                vessel2 = np.loadtxt(name1,delimiter=',')
                
                pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
                
                flowrate1=np.delete(vessel1, [0,2,3], 1)
                flowrate2=np.delete(vessel2, [0,2,3], 1)
                
                
                age=np.tile(age1,(len(flowrate1),1))
            
                
                target=np.array([0,0,1,0,0,0,0,0,0])
        
                    
                feature1=tf.concat([feature1,pressure1],axis=0)
                feature2=tf.concat([feature2,pressure2],axis=0)
                feature3=tf.concat([feature3,flowrate1],axis=0)
                feature4=tf.concat([feature4,flowrate2],axis=0)
                feature5=tf.concat([feature5,age],axis=0)
                target_dataset=tf.concat([target_dataset,target],axis=0)

            path="..\Snaps\CoW_"+str(j)+"_ICAR"+str(i)+"0.csv"
            if os.path.isfile(path):
                name1="..\Snaps\CoW_"+str(j)+"_ICAR00.csv"
                vessel1 = np.loadtxt(name1,delimiter=',')
                name1="..\Snaps\CoW_"+str(j)+"_ICAR01.csv"
                vessel2 = np.loadtxt(name1,delimiter=',')
                
                pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
                
                flowrate1=np.delete(vessel1, [0,2,3], 1)
                flowrate2=np.delete(vessel2, [0,2,3], 1)
                
                
                age=np.tile(age1,(len(flowrate1),1))
            
                
                target=np.array([0,1,0,0,0,0,0,0,0])
        
                    
                feature1=tf.concat([feature1,pressure1],axis=0)
                feature2=tf.concat([feature2,pressure2],axis=0)
                feature3=tf.concat([feature3,flowrate1],axis=0)
                feature4=tf.concat([feature4,flowrate2],axis=0)
                feature5=tf.concat([feature5,age],axis=0)
                target_dataset=tf.concat([target_dataset,target],axis=0)
                
            path="..\Snaps\CoW_"+str(j)+"_ICAR2"+str(i)+"0.csv"
            if os.path.isfile(path):
                name1="..\Snaps\CoW_"+str(j)+"_ICAR200.csv"
                vessel1 = np.loadtxt(name1,delimiter=',')
                name1="..\Snaps\CoW_"+str(j)+"_ICAR201.csv"
                vessel2 = np.loadtxt(name1,delimiter=',')
                
                pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
                
                flowrate1=np.delete(vessel1, [0,2,3], 1)
                flowrate2=np.delete(vessel2, [0,2,3], 1)
                
                
                age=np.tile(age1,(len(flowrate1),1))
            
                
                target=np.array([0,1,0,0,0,0,0,0,0])
        
                    
                feature1=tf.concat([feature1,pressure1],axis=0)
                feature2=tf.concat([feature2,pressure2],axis=0)
                feature3=tf.concat([feature3,flowrate1],axis=0)
                feature4=tf.concat([feature4,flowrate2],axis=0)
                feature5=tf.concat([feature5,age],axis=0)
                target_dataset=tf.concat([target_dataset,target],axis=0)
                
            path="..\Snaps\CoW_"+str(j)+"_PCAL"+str(i)+"0.csv"
            if os.path.isfile(path):
                name1="..\Snaps\CoW_"+str(j)+"_PCAL00.csv"
                vessel1 = np.loadtxt(name1,delimiter=',')
                name1="..\Snaps\CoW_"+str(j)+"_PCAL01.csv"
                vessel2 = np.loadtxt(name1,delimiter=',')
                
                pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
                
                flowrate1=np.delete(vessel1, [0,2,3], 1)
                flowrate2=np.delete(vessel2, [0,2,3], 1)
                
                
                age=np.tile(age1,(len(flowrate1),1))
            
                
                target=np.array([0,0,1,0,0,0,0,0,0])
        
                    
                feature1=tf.concat([feature1,pressure1],axis=0)
                feature2=tf.concat([feature2,pressure2],axis=0)
                feature3=tf.concat([feature3,flowrate1],axis=0)
                feature4=tf.concat([feature4,flowrate2],axis=0)
                feature5=tf.concat([feature5,age],axis=0)
                target_dataset=tf.concat([target_dataset,target],axis=0)
            """  
            path="..\Snaps\CoW_"+str(j)+"_PCAL2"+str(i)+"0.csv"
            if os.path.isfile(path):
                name1="..\Snaps\CoW_"+str(j)+"_PCAL200.csv"
                vessel1 = np.loadtxt(name1,delimiter=',')
                name1="..\Snaps\CoW_"+str(j)+"_PCAL201.csv"
                vessel2 = np.loadtxt(name1,delimiter=',')
                
                pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
                
                flowrate1=np.delete(vessel1, [0,2,3], 1)
                flowrate2=np.delete(vessel2, [0,2,3], 1)
                
                
                age=np.tile(age1,(len(flowrate1),1))
            
                
                target=np.array([0,0,1,0,0,0,0,0,0])
        
                    
                feature1=tf.concat([feature1,pressure1],axis=0)
                feature2=tf.concat([feature2,pressure2],axis=0)
                feature3=tf.concat([feature3,flowrate1],axis=0)
                feature4=tf.concat([feature4,flowrate2],axis=0)
                feature5=tf.concat([feature5,age],axis=0)
                target_dataset=tf.concat([target_dataset,target],axis=0)
            """
            path="..\Snaps\CoW_"+str(j)+"_PCAR"+str(i)+"0.csv"
            if os.path.isfile(path):
                name1="..\Snaps\CoW_"+str(j)+"_PCAR00.csv"
                vessel1 = np.loadtxt(name1,delimiter=',')
                name1="..\Snaps\CoW_"+str(j)+"_PCAR01.csv"
                vessel2 = np.loadtxt(name1,delimiter=',')
                
                pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
                
                flowrate1=np.delete(vessel1, [0,2,3], 1)
                flowrate2=np.delete(vessel2, [0,2,3], 1)
                
                
                age=np.tile(age1,(len(flowrate1),1))
            
                
                target=np.array([0,1,0,0,0,0,0,0,0])
        
                    
                feature1=tf.concat([feature1,pressure1],axis=0)
                feature2=tf.concat([feature2,pressure2],axis=0)
                feature3=tf.concat([feature3,flowrate1],axis=0)
                feature4=tf.concat([feature4,flowrate2],axis=0)
                feature5=tf.concat([feature5,age],axis=0)
                target_dataset=tf.concat([target_dataset,target],axis=0)
            """   
            path="..\Snaps\CoW_"+str(j)+"_PCAR2"+str(i)+"0.csv"
            if os.path.isfile(path):
                name1="..\Snaps\CoW_"+str(j)+"_PCAR200.csv"
                vessel1 = np.loadtxt(name1,delimiter=',')
                name1="..\Snaps\CoW_"+str(j)+"_PCAR201.csv"
                vessel2 = np.loadtxt(name1,delimiter=',')
                
                pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
                
                flowrate1=np.delete(vessel1, [0,2,3], 1)
                flowrate2=np.delete(vessel2, [0,2,3], 1)
                
                
                age=np.tile(age1,(len(flowrate1),1))
            
                
                target=np.array([0,1,0,0,0,0,0,0,0])
        
                    
                feature1=tf.concat([feature1,pressure1],axis=0)
                feature2=tf.concat([feature2,pressure2],axis=0)
                feature3=tf.concat([feature3,flowrate1],axis=0)
                feature4=tf.concat([feature4,flowrate2],axis=0)
                feature5=tf.concat([feature5,age],axis=0)
                target_dataset=tf.concat([target_dataset,target],axis=0)
            """
            path="..\Snaps\CoW_"+str(j)+"_VAL"+str(i)+"0.csv"
            if os.path.isfile(path):
                name1="..\Snaps\CoW_"+str(j)+"_VAL00.csv"
                vessel1 = np.loadtxt(name1,delimiter=',')
                name1="..\Snaps\CoW_"+str(j)+"_VAL01.csv"
                vessel2 = np.loadtxt(name1,delimiter=',')
                
                pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
                
                flowrate1=np.delete(vessel1, [0,2,3], 1)
                flowrate2=np.delete(vessel2, [0,2,3], 1)
                
                
                age=np.tile(age1,(len(flowrate1),1))
            
                
                target=np.array([0,0,1,0,0,0,0,0,0])
        
                    
                feature1=tf.concat([feature1,pressure1],axis=0)
                feature2=tf.concat([feature2,pressure2],axis=0)
                feature3=tf.concat([feature3,flowrate1],axis=0)
                feature4=tf.concat([feature4,flowrate2],axis=0)
                feature5=tf.concat([feature5,age],axis=0)
                target_dataset=tf.concat([target_dataset,target],axis=0)
                
            path="..\Snaps\CoW_"+str(j)+"_VAR"+str(i)+"0.csv"
            if os.path.isfile(path):
                name1="..\Snaps\CoW_"+str(j)+"_VAR00.csv"
                vessel1 = np.loadtxt(name1,delimiter=',')
                name1="..\Snaps\CoW_"+str(j)+"_VAR01.csv"
                vessel2 = np.loadtxt(name1,delimiter=',')
                
                pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
                
                flowrate1=np.delete(vessel1, [0,2,3], 1)
                flowrate2=np.delete(vessel2, [0,2,3], 1)
                
                
                age=np.tile(age1,(len(flowrate1),1))
            
                
                target=np.array([0,1,0,0,0,0,0,0,0])
        
                    
                feature1=tf.concat([feature1,pressure1],axis=0)
                feature2=tf.concat([feature2,pressure2],axis=0)
                feature3=tf.concat([feature3,flowrate1],axis=0)
                feature4=tf.concat([feature4,flowrate2],axis=0)
                feature5=tf.concat([feature5,age],axis=0)
                target_dataset=tf.concat([target_dataset,target],axis=0)
            

        else:
            if i in ACALvessels:
                path="..\Snaps\CoW_"+str(j)+"_ACAL"+str(i)+"0.csv"
                if os.path.isfile(path):
                    name1="..\Snaps\CoW_"+str(j)+"_ACAL"+str(i)+"0.csv"
                    vessel1 = np.loadtxt(name1,delimiter=',')
                    name1="..\Snaps\CoW_"+str(j)+"_ACAL"+str(i)+"1.csv"
                    vessel2 = np.loadtxt(name1,delimiter=',')
                
                    if j==20 or j ==21 or j ==22 or j==23:
                        pressure1=np.square(np.delete(vessel1, [0,1,2,3,5], 1)/beta+np.sqrt(A0))
                        pressure2=np.square(np.delete(vessel2, [0,1,2,3,5], 1)/beta+np.sqrt(A0))
                        
                        flowrate1=np.delete(vessel1, [0,1,3,4,5], 1)
                        flowrate2=np.delete(vessel2, [0,1,3,4,5], 1)
                    else:
                        pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                        pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
                        
                        flowrate1=np.delete(vessel1, [0,2,3], 1)
                        flowrate2=np.delete(vessel2, [0,2,3], 1)
                    
                    
                    age=np.tile(age1,(len(flowrate1),1))
                
                
                    target=np.array([0,0,0,0,1,0,0,0,0])
            
                        

                    feature1=tf.concat([feature1,pressure1],axis=0)
                    feature2=tf.concat([feature2,pressure2],axis=0)
                    feature3=tf.concat([feature3,flowrate1],axis=0)
                    feature4=tf.concat([feature4,flowrate2],axis=0)
                    feature5=tf.concat([feature5,age],axis=0)
                    target_dataset=tf.concat([target_dataset,target],axis=0)
                    
            elif i in ACARvessels:
                path="..\Snaps\CoW_"+str(j)+"_ACAR"+str(i)+"0.csv"
                if os.path.isfile(path):
                    name1="..\Snaps\CoW_"+str(j)+"_ACAR"+str(i)+"0.csv"
                    vessel1 = np.loadtxt(name1,delimiter=',')
                    name1="..\Snaps\CoW_"+str(j)+"_ACAR"+str(i)+"1.csv"
                    vessel2 = np.loadtxt(name1,delimiter=',')
                
    
                    if j==20 or j ==21 or j ==22 or j==23:
                        pressure1=np.square(np.delete(vessel1, [0,1,2,3,5], 1)/beta+np.sqrt(A0))
                        pressure2=np.square(np.delete(vessel2, [0,1,2,3,5], 1)/beta+np.sqrt(A0))
                        
                        flowrate1=np.delete(vessel1, [0,1,3,4,5], 1)
                        flowrate2=np.delete(vessel2, [0,1,3,4,5], 1)
                    else:
                        pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                        pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
                        
                        flowrate1=np.delete(vessel1, [0,2,3], 1)
                        flowrate2=np.delete(vessel2, [0,2,3], 1)
                    

                    
                    age=np.tile(age1,(len(flowrate1),1))
                
                    
                    target=np.array([0,0,0,1,0,0,0,0,0])
            
                        
                    feature1=tf.concat([feature1,pressure1],axis=0)
                    feature2=tf.concat([feature2,pressure2],axis=0)
                    feature3=tf.concat([feature3,flowrate1],axis=0)
                    feature4=tf.concat([feature4,flowrate2],axis=0)
                    feature5=tf.concat([feature5,age],axis=0)
                    target_dataset=tf.concat([target_dataset,target],axis=0)
                
            elif i in MCARvessels:
                path="..\Snaps\CoW_"+str(j)+"_MCAR"+str(i)+"0.csv"
                if os.path.isfile(path):
                    name1="..\Snaps\CoW_"+str(j)+"_MCAR"+str(i)+"0.csv"
                    vessel1 = np.loadtxt(name1,delimiter=',')
                    name1="..\Snaps\CoW_"+str(j)+"_MCAR"+str(i)+"1.csv"
                    vessel2 = np.loadtxt(name1,delimiter=',')
                
    
            
                    if j==20 or j ==21 or j ==22 or j==23:
                        pressure1=np.square(np.delete(vessel1, [0,1,2,3,5], 1)/beta+np.sqrt(A0))
                        pressure2=np.square(np.delete(vessel2, [0,1,2,3,5], 1)/beta+np.sqrt(A0))
                        
                        flowrate1=np.delete(vessel1, [0,1,3,4,5], 1)
                        flowrate2=np.delete(vessel2, [0,1,3,4,5], 1)
                    else:
                        pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                        pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
                        
                        flowrate1=np.delete(vessel1, [0,2,3], 1)
                        flowrate2=np.delete(vessel2, [0,2,3], 1)
                    
                    
                    age=np.tile(age1,(len(flowrate1),1))
                
                    if i==68:
                        target=np.array([0,1,0,0,0,0,0,0,0])
                    else:
                        target=np.array([0,0,0,0,0,1,0,0,0])
            
                        
                    feature1=tf.concat([feature1,pressure1],axis=0)
                    feature2=tf.concat([feature2,pressure2],axis=0)
                    feature3=tf.concat([feature3,flowrate1],axis=0)
                    feature4=tf.concat([feature4,flowrate2],axis=0)
                    feature5=tf.concat([feature5,age],axis=0)
                    target_dataset=tf.concat([target_dataset,target],axis=0)
                    
            elif i in MCALvessels:
                path="..\Snaps\CoW_"+str(j)+"_MCAL"+str(i)+"0.csv"
                if os.path.isfile(path):
                    name1="..\Snaps\CoW_"+str(j)+"_MCAL"+str(i)+"0.csv"
                    vessel1 = np.loadtxt(name1,delimiter=',')
                    name1="..\Snaps\CoW_"+str(j)+"_MCAL"+str(i)+"1.csv"
                    vessel2 = np.loadtxt(name1,delimiter=',')
                
    
            
                    if j==20 or j ==21 or j ==22 or j==23:
                        pressure1=np.square(np.delete(vessel1, [0,1,2,3,5], 1)/beta+np.sqrt(A0))
                        pressure2=np.square(np.delete(vessel2, [0,1,2,3,5], 1)/beta+np.sqrt(A0))
                        
                        flowrate1=np.delete(vessel1, [0,1,3,4,5], 1)
                        flowrate2=np.delete(vessel2, [0,1,3,4,5], 1)
                    else:
                        pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                        pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
                        
                        flowrate1=np.delete(vessel1, [0,2,3], 1)
                        flowrate2=np.delete(vessel2, [0,2,3], 1)
                    
                    
                    age=np.tile(age1,(len(flowrate1),1))
                
                    
                    if i==50:
                        target=np.array([0,0,1,0,0,0,0,0,0])
                    else:
                        target=np.array([0,0,0,0,0,0,1,0,0])
            
                        
                    feature1=tf.concat([feature1,pressure1],axis=0)
                    feature2=tf.concat([feature2,pressure2],axis=0)
                    feature3=tf.concat([feature3,flowrate1],axis=0)
                    feature4=tf.concat([feature4,flowrate2],axis=0)
                    feature5=tf.concat([feature5,age],axis=0)
                    target_dataset=tf.concat([target_dataset,target],axis=0)
                    
            elif i in PCARvessels:
                path="..\Snaps\CoW_"+str(j)+"_PCAR"+str(i)+"0.csv"
                if os.path.isfile(path):
                    name1="..\Snaps\CoW_"+str(j)+"_PCAR"+str(i)+"0.csv"
                    vessel1 = np.loadtxt(name1,delimiter=',')
                    name1="..\Snaps\CoW_"+str(j)+"_PCAR"+str(i)+"1.csv"
                    vessel2 = np.loadtxt(name1,delimiter=',')
                
    
            
                    if j==20 or j ==21 or j ==22 or j==23:
                        pressure1=np.square(np.delete(vessel1, [0,1,2,3,5], 1)/beta+np.sqrt(A0))
                        pressure2=np.square(np.delete(vessel2, [0,1,2,3,5], 1)/beta+np.sqrt(A0))
                        
                        flowrate1=np.delete(vessel1, [0,1,3,4,5], 1)
                        flowrate2=np.delete(vessel2, [0,1,3,4,5], 1)
                    else:
                        pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                        pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
                        
                        flowrate1=np.delete(vessel1, [0,2,3], 1)
                        flowrate2=np.delete(vessel2, [0,2,3], 1)
                    
                    
                    age=np.tile(age1,(len(flowrate1),1))
                
                    
                    target=np.array([0,0,0,0,0,0,0,1,0])
            
                        
                    feature1=tf.concat([feature1,pressure1],axis=0)
                    feature2=tf.concat([feature2,pressure2],axis=0)
                    feature3=tf.concat([feature3,flowrate1],axis=0)
                    feature4=tf.concat([feature4,flowrate2],axis=0)
                    feature5=tf.concat([feature5,age],axis=0)
                    target_dataset=tf.concat([target_dataset,target],axis=0)
            
            elif i in PCALvessels:
                path="..\Snaps\CoW_"+str(j)+"_PCAL"+str(i)+"0.csv"
                if os.path.isfile(path):
                    name1="..\Snaps\CoW_"+str(j)+"_PCAL"+str(i)+"0.csv"
                    vessel1 = np.loadtxt(name1,delimiter=',')
                    name1="..\Snaps\CoW_"+str(j)+"_PCAL"+str(i)+"1.csv"
                    vessel2 = np.loadtxt(name1,delimiter=',')
                
    
                    if j==20 or j ==21 or j ==22 or j==23:
                        pressure1=np.square(np.delete(vessel1, [0,1,2,3,5], 1)/beta+np.sqrt(A0))
                        pressure2=np.square(np.delete(vessel2, [0,1,2,3,5], 1)/beta+np.sqrt(A0))
                        
                        flowrate1=np.delete(vessel1, [0,1,3,4,5], 1)
                        flowrate2=np.delete(vessel2, [0,1,3,4,5], 1)
                    else:
                        pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                        pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
                        
                        flowrate1=np.delete(vessel1, [0,2,3], 1)
                        flowrate2=np.delete(vessel2, [0,2,3], 1)
                                   
                    
                    age=np.tile(age1,(len(flowrate1),1))
                
                    
                    
                    target=np.array([0,0,0,0,0,0,0,0,1])
            
                        
                    feature1=tf.concat([feature1,pressure1],axis=0)
                    feature2=tf.concat([feature2,pressure2],axis=0)
                    feature3=tf.concat([feature3,flowrate1],axis=0)
                    feature4=tf.concat([feature4,flowrate2],axis=0)
                    feature5=tf.concat([feature5,age],axis=0)
                    target_dataset=tf.concat([target_dataset,target],axis=0)
                

N_t=len(flowrate1)

A_mean=np.mean(np.vstack([feature1,feature2]))
A_std=np.std(np.vstack([feature1,feature2]))

V_mean=np.mean(np.vstack([feature3,feature4]))
V_std=np.std(np.vstack([feature3,feature4]))

Age_mean=np.mean(feature5)
Age_std=np.std(feature5)

feature1=(feature1-A_mean)/A_std
feature2=(feature2-A_mean)/A_std
feature3=(feature3-V_mean)/V_std
feature4=(feature4-V_mean)/V_std
feature5=(feature5-Age_mean)/Age_std
data=np.hstack([feature3,feature4,feature5])
target_dataset=np.hstack([target_dataset])
#### output data #####

y_dataset1 = []
x_dataset1 = []
counter=0
counterr=0
N_outlet=len(feature1)/N_t
for j in range(0,int(N_outlet)):
    x_dataset1.append(data[N_t*counter:N_t*(counter+1),:])
    y_dataset1.append(target_dataset[counterr:counterr+9])
    counter=counter+1
    counterr=counterr+9
    




y_dataset2= np.array(y_dataset1)
x_dataset2= np.array(x_dataset1)

y_dataset=np.vstack([y_dataset2,y_dataset2,y_dataset2])
x_dataset=np.vstack([x_dataset2,x_dataset2,x_dataset2])

sample=x_dataset.shape[1]
for i in range(0,x_dataset.shape[0]-x_dataset2.shape[0]):
    amp1=np.max(x_dataset[x_dataset2.shape[0]+i,:,0])
    amp2=np.max(x_dataset[x_dataset2.shape[0]+i,:,1])
    noise1 = ((amp1*random.randint(0, 20))/100/1000)*np.asarray(random.sample(range(0,1000),sample)) 
    noise2 = ((amp2*random.randint(0, 20))/100/1000)*np.asarray(random.sample(range(0,1000),sample)) 
    x_dataset[x_dataset2.shape[0]+i,:,0]=x_dataset[x_dataset2.shape[0]+i,:,0]+noise1
    x_dataset[x_dataset2.shape[0]+i,:,1]=x_dataset[x_dataset2.shape[0]+i,:,1]+noise2


train_size=int(train_size*3)
test_size=len(x_dataset)-train_size


indices = np.random.permutation(x_dataset.shape[0])
training_idx, test_idx = indices[:int(train_size)], indices[int(train_size):]
x_train, x_test = x_dataset[training_idx,:], x_dataset[test_idx,:]
y_train, y_test = y_dataset[training_idx,:], y_dataset[test_idx,:]


#### model and training #####    
model = tf.keras.Sequential()
# convolutional layer
model.add(tf.keras.layers.Conv1D(60, kernel_size=5, strides=1, padding='valid', activation='relu', input_shape=(N_t,n_features)))
model.add(tf.keras.layers.Conv1D(60, kernel_size=5, strides=1, padding='valid', activation='relu'))
model.add(tf.keras.layers.MaxPool1D(pool_size=2))
model.add(tf.keras.layers.Conv1D(40, kernel_size=5, strides=1, padding='valid', activation='relu'))
model.add(tf.keras.layers.Conv1D(40, kernel_size=5, strides=1, padding='valid', activation='relu'))
model.add(tf.keras.layers.MaxPool1D(pool_size=2))
model.add(tf.keras.layers.Conv1D(20, kernel_size=5 , strides=1, padding='valid', activation='relu'))
model.add(tf.keras.layers.Conv1D(20, kernel_size=5 , strides=1, padding='valid', activation='relu'))
model.add(tf.keras.layers.MaxPool1D(pool_size=2))
model.add(tf.keras.layers.Conv1D(10, kernel_size=5 , strides=1, padding='valid', activation='relu'))
model.add(tf.keras.layers.Conv1D(5, kernel_size=5 , strides=1, padding='valid', activation='relu'))
# flatten output of conv
model.add(tf.keras.layers.Flatten())
# hidden layer
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(20, activation='relu'))
model.add(tf.keras.layers.Dense(20, activation='relu'))

# output layer
model.add(tf.keras.layers.Dense(9, activation='softmax'))


if train_size==5000:
    class CustomCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs.get('val_accuracy')>0.91:
                self.model.stop_training = True
elif train_size==6000:
    class CustomCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs.get('val_accuracy')>0.95:
                self.model.stop_training = True
elif train_size==7000:
    class CustomCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs.get('val_accuracy')>0.95:
                self.model.stop_training = True
elif train_size==8000:
    class CustomCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs.get('val_accuracy')>0.96:
                self.model.stop_training = True
else :
    class CustomCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs.get('val_accuracy')>0.91 or  logs.get('accuracy')>0.95:
                self.model.stop_training = True
            
log_dir = "logs/fit1/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


model.compile(loss='BinaryCrossentropy', metrics=['accuracy'], optimizer='adam')
model.fit(x_train, y_train, batch_size=256,callbacks=[CustomCallback()], epochs=15000,validation_data=(x_test,y_test))


A=model.predict(x_train)
predict=np.around(A)

pred=np.zeros(train_size)
true=np.ones(train_size)

for i in range(0,train_size):
    if np.array_equal(predict[i,:],y_train[i,:]):
       pred[i]=1   


F1_train=f1_score(true, pred, average='micro', zero_division=0)

model.evaluate(x_test, y_test)

A=model.predict(x_test)
predict=np.around(A)

pred=np.zeros(len(x_test))
true=np.ones(len(x_test))

for i in range(0,len(x_test)):
    if np.array_equal(predict[i,:],y_test[i,:]):
       pred[i]=1  

F1_test=f1_score(y_test, predict, average='micro', zero_division=0)
name='model_location_vel_noise_'+str(train_size)+'.h5'
model.save(name)

    

cma=np.zeros((9,9))
for i in range(0,len(A)):
    for j in range(0,9):
        for k in range(0,9):
            if predict[i,j]==1 and y_test[i,k]==1:
                cma[k,j]=cma[k,j]+1
                
for i in range(0,9):
    number=np.sum(cma[i,:])
    cma[i,:]=cma[i,:]/number
            
cma=np.round(cma,decimals=2)

print(F1_test)

fpr, tpr,_ = metrics.roc_curve(y_test[:,i], A[:,i],pos_label=1,drop_intermediate=False)
name='fpr_region_vel_noise_'+str(train_size)+'_'
np.save(name,fpr)

name='tpr_region_vel_noise_'+str(train_size)+'_'
np.save(name,tpr)

name='cma_region_vel_noise_'+str(train_size)+'_'
np.save(name,cma)

name='f1test_region_vel_noise_'+str(train_size)+'_'
np.save(name,F1_test)

name='P_nor_region_vel_noise_'+str(train_size)+'_'
np.save(name,A_mean)

name='P_std_region_vel_noise_'+str(train_size)+'_'
np.save(name,A_std)

name='V_nor_region_vel_noise_'+str(train_size)+'_'
np.save(name,V_mean)

name='V_std_region_vel_noise_'+str(train_size)+'_'
np.save(name,V_std)

name='Age_nor_region_vel_noise_'+str(train_size)+'_'
np.save(name,Age_mean)

name='Age_std_region_vel_noise_'+str(train_size)+'_'
np.save(name,Age_std)