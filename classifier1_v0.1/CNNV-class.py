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

DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)

N_mu=500
numbermodes=1
n_features = 3
train_size=8000
mode=0 #both pressure vel, 1 only velocity,2 only pressure
Noise=0 #noise %

a = np.ones(1)*0.2802 #
b =  np.ones(1)*(-0.5053)*1000 #m-1
c =  np.ones(1)*0.1324 #
d =  np.ones(1)*(-0.01114)*1000 #m-1
E=1.6*1e6

ACALvessels=np.loadtxt('..\samples\CoW_0_ACAL.csv',delimiter=',')
MCALvessels=np.loadtxt('..\samples\CoW_0_MCAL.csv',delimiter=',')
PCALvessels=np.loadtxt('..\samples\CoW_0_PCAL.csv',delimiter=',')
ACARvessels=np.loadtxt('..\samples\CoW_0_ACAR.csv',delimiter=',')
MCARvessels=np.loadtxt('..\samples\CoW_0_MCAR.csv',delimiter=',')
PCARvessels=np.loadtxt('..\samples\CoW_0_PCAR.csv',delimiter=',')

#### input data #####            
for j in range(0,N_mu):
    for i in range(0,130):
        name='..\samples\parameters'+str(j)+'.npy'
        if os.path.isfile(name):
            age1=np.load(name)
        else:
            age1=np.linspace(30, 80, num=10)[(j%10)]
            
        A0=ata.CCA(age1)
        r=np.sqrt(A0/np.pi)
        h=r*(a*np.exp(b*r)+c*np.exp(d*r))
        beta=(4/3)*((np.sqrt(np.pi)*E*h)/(A0))
        
        if i==0:
            path="..\samples\CoW_"+str(j)+"_"+str(i)+"0.csv"
            if os.path.isfile(path):
                name1="..\samples\CoW_"+str(j)+"_00.csv"
                vessel1 = np.loadtxt(name1,delimiter=',')
                name1="..\samples\CoW_"+str(j)+"_01.csv"
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
                    
            path="..\samples\CoW_"+str(j)+"_ACOR"+str(i)+"0.csv"
            if os.path.isfile(path):
                name1="..\samples\CoW_"+str(j)+"_ACOR00.csv"
                vessel1 = np.loadtxt(name1,delimiter=',')
                name1="..\samples\CoW_"+str(j)+"_ACOR01.csv"
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
                    
            path="..\samples\CoW_"+str(j)+"_ACOL"+str(i)+"0.csv"
            if os.path.isfile(path):
                name1="..\samples\CoW_"+str(j)+"_ACOL00.csv"
                vessel1 = np.loadtxt(name1,delimiter=',')
                name1="..\samples\CoW_"+str(j)+"_ACOL01.csv"
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
                
            

            path="..\samples\CoW_"+str(j)+"_ICAL"+str(i)+"0.csv"
            if os.path.isfile(path):
                name1="..\samples\CoW_"+str(j)+"_ICAL00.csv"
                vessel1 = np.loadtxt(name1,delimiter=',')
                name1="..\samples\CoW_"+str(j)+"_ICAL01.csv"
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
                
                
            path="..\samples\CoW_"+str(j)+"_ICAL2"+str(i)+"0.csv"
            if os.path.isfile(path):
                name1="..\samples\CoW_"+str(j)+"_ICAL200.csv"
                vessel1 = np.loadtxt(name1,delimiter=',')
                name1="..\samples\CoW_"+str(j)+"_ICAL201.csv"
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

            path="..\samples\CoW_"+str(j)+"_ICAR"+str(i)+"0.csv"
            if os.path.isfile(path):
                name1="..\samples\CoW_"+str(j)+"_ICAR00.csv"
                vessel1 = np.loadtxt(name1,delimiter=',')
                name1="..\samples\CoW_"+str(j)+"_ICAR01.csv"
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
                
            path="..\samples\CoW_"+str(j)+"_ICAR2"+str(i)+"0.csv"
            if os.path.isfile(path):
                name1="..\samples\CoW_"+str(j)+"_ICAR200.csv"
                vessel1 = np.loadtxt(name1,delimiter=',')
                name1="..\samples\CoW_"+str(j)+"_ICAR201.csv"
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
                
            path="..\samples\CoW_"+str(j)+"_PCAL"+str(i)+"0.csv"
            if os.path.isfile(path):
                name1="..\samples\CoW_"+str(j)+"_PCAL00.csv"
                vessel1 = np.loadtxt(name1,delimiter=',')
                name1="..\samples\CoW_"+str(j)+"_PCAL01.csv"
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

            path="..\samples\CoW_"+str(j)+"_PCAR"+str(i)+"0.csv"
            if os.path.isfile(path):
                name1="..\samples\CoW_"+str(j)+"_PCAR00.csv"
                vessel1 = np.loadtxt(name1,delimiter=',')
                name1="..\samples\CoW_"+str(j)+"_PCAR01.csv"
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

            path="..\samples\CoW_"+str(j)+"_VAL"+str(i)+"0.csv"
            if os.path.isfile(path):
                name1="..\samples\CoW_"+str(j)+"_VAL00.csv"
                vessel1 = np.loadtxt(name1,delimiter=',')
                name1="..\samples\CoW_"+str(j)+"_VAL01.csv"
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
                
            path="..\samples\CoW_"+str(j)+"_VAR"+str(i)+"0.csv"
            if os.path.isfile(path):
                name1="..\samples\CoW_"+str(j)+"_VAR00.csv"
                vessel1 = np.loadtxt(name1,delimiter=',')
                name1="..\samples\CoW_"+str(j)+"_VAR01.csv"
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
                path="..\samples\CoW_"+str(j)+"_ACAL"+str(i)+"0.csv"
                if os.path.isfile(path):
                    name1="..\samples\CoW_"+str(j)+"_ACAL"+str(i)+"0.csv"
                    vessel1 = np.loadtxt(name1,delimiter=',')
                    name1="..\samples\CoW_"+str(j)+"_ACAL"+str(i)+"1.csv"
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
                path="..\samples\CoW_"+str(j)+"_ACAR"+str(i)+"0.csv"
                if os.path.isfile(path):
                    name1="..\samples\CoW_"+str(j)+"_ACAR"+str(i)+"0.csv"
                    vessel1 = np.loadtxt(name1,delimiter=',')
                    name1="..\samples\CoW_"+str(j)+"_ACAR"+str(i)+"1.csv"
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
                path="..\samples\CoW_"+str(j)+"_MCAR"+str(i)+"0.csv"
                if os.path.isfile(path):
                    name1="..\samples\CoW_"+str(j)+"_MCAR"+str(i)+"0.csv"
                    vessel1 = np.loadtxt(name1,delimiter=',')
                    name1="..\samples\CoW_"+str(j)+"_MCAR"+str(i)+"1.csv"
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
                path="..\samples\CoW_"+str(j)+"_MCAL"+str(i)+"0.csv"
                if os.path.isfile(path):
                    name1="..\samples\CoW_"+str(j)+"_MCAL"+str(i)+"0.csv"
                    vessel1 = np.loadtxt(name1,delimiter=',')
                    name1="..\samples\CoW_"+str(j)+"_MCAL"+str(i)+"1.csv"
                    vessel2 = np.loadtxt(name1,delimiter=',')
                
    
            

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
                path="..\samples\CoW_"+str(j)+"_PCAR"+str(i)+"0.csv"
                if os.path.isfile(path):
                    name1="..\samples\CoW_"+str(j)+"_PCAR"+str(i)+"0.csv"
                    vessel1 = np.loadtxt(name1,delimiter=',')
                    name1="..\samples\CoW_"+str(j)+"_PCAR"+str(i)+"1.csv"
                    vessel2 = np.loadtxt(name1,delimiter=',')
                
    
            

                    age=np.tile(age1,(len(flowrate1),1))
                
                    
                    target=np.array([0,0,0,0,0,0,0,1,0])
            
                        
                    feature1=tf.concat([feature1,pressure1],axis=0)
                    feature2=tf.concat([feature2,pressure2],axis=0)
                    feature3=tf.concat([feature3,flowrate1],axis=0)
                    feature4=tf.concat([feature4,flowrate2],axis=0)
                    feature5=tf.concat([feature5,age],axis=0)
                    target_dataset=tf.concat([target_dataset,target],axis=0)
            
            elif i in PCALvessels:
                path="..\samples\CoW_"+str(j)+"_PCAL"+str(i)+"0.csv"
                if os.path.isfile(path):
                    name1="..\samples\CoW_"+str(j)+"_PCAL"+str(i)+"0.csv"
                    vessel1 = np.loadtxt(name1,delimiter=',')
                    name1="..\samples\CoW_"+str(j)+"_PCAL"+str(i)+"1.csv"
                    vessel2 = np.loadtxt(name1,delimiter=',')
                

                                   
                    
                    age=np.tile(age1,(len(flowrate1),1))
                
                    
                    
                    target=np.array([0,0,0,0,0,0,0,0,1])
            
                        
                    feature1=tf.concat([feature1,pressure1],axis=0)
                    feature2=tf.concat([feature2,pressure2],axis=0)
                    feature3=tf.concat([feature3,flowrate1],axis=0)
                    feature4=tf.concat([feature4,flowrate2],axis=0)
                    feature5=tf.concat([feature5,age],axis=0)
                    target_dataset=tf.concat([target_dataset,target],axis=0)
                

N_t=len(flowrate1)

P_mean=np.mean(np.vstack([feature1,feature2]))
P_std=np.std(np.vstack([feature1,feature2]))

V_mean=np.mean(np.vstack([feature3,feature4]))
V_std=np.std(np.vstack([feature3,feature4]))

Age_mean=np.mean(feature5)
Age_std=np.std(feature5)

feature1=(feature1-P_mean)/P_std
feature2=(feature2-P_mean)/P_std
feature3=(feature3-V_mean)/V_std
feature4=(feature4-V_mean)/V_std
if mode==0:
    data=np.hstack([feature1,feature2,feature3,feature4])
elif mode==1:
    data=np.hstack([feature3,feature4])
elif mode==2:
    data=np.hstack([feature1,feature2])
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
    




y_dataset= np.array(y_dataset1)
x_dataset= np.array(x_dataset1)


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


log_dir = "logs/fit1/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


model.compile(loss='BinaryCrossentropy', metrics=['accuracy'], optimizer='adam')
model.fit(x_train, y_train, batch_size=256, epochs=100000)


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
model.save('model_location_vel.h5')

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 8),dpi=600)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    #plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=12, fontname="Arial")
        plt.yticks(tick_marks, target_names, fontsize=12, fontname="Arial")

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label', fontsize=34, fontname="Arial")
    plt.xlabel('Predicted label', fontsize=34, fontname="Arial")
    plt.show()
    

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

plot_confusion_matrix(cm           = cma, 
                      normalize    = False,
                      target_names = ['Healty', 'LVO R', 'LVO L' , 'ACA R' , 'ACA L' , 'MCA R', 'MCA L', 'PCA R', 'PCA L'],
                      title        = "")
print(F1_test)

fpr, tpr,_ = metrics.roc_curve(y_test[:,i], A[:,i],pos_label=1,drop_intermediate=False)
