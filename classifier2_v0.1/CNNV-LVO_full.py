import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import f1_score
import artery_area as ata
import datetime
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

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


DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)
counterrr=0
mode=0 #both pressure vel, 1 only velocity,2 only pressure
for train_size in [200]:
    N_mu=500
    n_features = 5
    
    
    a = np.ones(1)*0.2802 #
    b =  np.ones(1)*(-0.5053)*1000 #m-1
    c =  np.ones(1)*0.1324 #
    d =  np.ones(1)*(-0.01114)*1000 #m-1
    E=0.8*1e6
    
    
    #### input data #####            
    for j in range(0,N_mu):
        i=0
        name='..\samples\Age'+str(j)+'.npy'
        if os.path.isfile(name):
            age1=np.load(name)
        else:
            age1=np.linspace(30, 80, num=10)[(j%10)]
            
        A0=ata.CCA(age1)
        r=np.sqrt(A0/np.pi)
        h=r*(a*np.exp(b*r)+c*np.exp(d*r))
        beta=(4/3)*((np.sqrt(np.pi)*E*h)/(A0))
        
        name1="..\samples\CoW_"+str(j)+"_01.csv"    
        path="..\samples\CoW_"+str(j)+"_ACOR"+str(i)+"0.csv"
        if os.path.isfile(path) and os.path.isfile(name1):
            name1="..\samples\CoW_"+str(j)+"_ACOR01.csv"
            vessel2 = np.loadtxt(name1,delimiter=',')
            name1="..\samples\CoW_"+str(j)+"_01.csv"
            vessel1 = np.loadtxt(name1,delimiter=',')
            
            if j==20 or j ==21 or j ==22 or j==23:
                pressure1=np.square(np.delete(vessel1, [0,1,2,3,5], 1)/beta+np.sqrt(A0))                
                flowrate1=np.delete(vessel1, [0,1,3,4,5], 1)
                
            else:
                pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                flowrate1=np.delete(vessel1, [0,2,3], 1)
            
            pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
            flowrate2=np.delete(vessel2, [0,2,3], 1)
            
            
            age=np.tile(age1,(len(flowrate1),1))
        
            
            target=np.array([1,0,0,0,0,0])
    
            if j==0:
                feature1=pressure1
                feature2=pressure2
                feature3=flowrate1
                feature4=flowrate2
                feature5=age
                target_dataset=target
            else:
                feature1=tf.concat([feature1,pressure1],axis=0)
                feature2=tf.concat([feature2,pressure2],axis=0)
                feature3=tf.concat([feature3,flowrate1],axis=0)
                feature4=tf.concat([feature4,flowrate2],axis=0)
                feature5=tf.concat([feature5,age],axis=0)
                target_dataset=tf.concat([target_dataset,target],axis=0)
        
        name1="..\samples\CoW_"+str(j)+"_00.csv"        
        path="..\samples\CoW_"+str(j)+"_ACOL"+str(i)+"0.csv"
        if os.path.isfile(path) and os.path.isfile(name1):
            name1="..\samples\CoW_"+str(j)+"_ACOL00.csv"
            vessel2 = np.loadtxt(name1,delimiter=',')
            name1="..\samples\CoW_"+str(j)+"_00.csv"
            vessel1 = np.loadtxt(name1,delimiter=',')
            
            if j==20 or j ==21 or j ==22 or j==23:
                pressure1=np.square(np.delete(vessel1, [0,1,2,3,5], 1)/beta+np.sqrt(A0))                
                flowrate1=np.delete(vessel1, [0,1,3,4,5], 1)
                
            else:
                pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                flowrate1=np.delete(vessel1, [0,2,3], 1)
            
            pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
            flowrate2=np.delete(vessel2, [0,2,3], 1)
            
            
            age=np.tile(age1,(len(flowrate1),1))
        
            
            target=np.array([1,0,0,0,0,0])
    
                
            feature1=tf.concat([feature1,pressure1],axis=0)
            feature2=tf.concat([feature2,pressure2],axis=0)
            feature3=tf.concat([feature3,flowrate1],axis=0)
            feature4=tf.concat([feature4,flowrate2],axis=0)
            feature5=tf.concat([feature5,age],axis=0)
            target_dataset=tf.concat([target_dataset,target],axis=0)
            
        
        name1="..\samples\CoW_"+str(j)+"_00.csv"
        path="..\samples\CoW_"+str(j)+"_ICAL"+str(i)+"0.csv"
        if os.path.isfile(path) and os.path.isfile(name1):
            name1="..\samples\CoW_"+str(j)+"_ICAL00.csv"
            vessel2 = np.loadtxt(name1,delimiter=',')
            name1="..\samples\CoW_"+str(j)+"_00.csv"
            vessel1 = np.loadtxt(name1,delimiter=',')
            

            pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
            flowrate1=np.delete(vessel1, [0,2,3], 1)
        
            pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
            flowrate2=np.delete(vessel2, [0,2,3], 1)
            
            
            age=np.tile(age1,(len(flowrate1),1))
        
            
            target=np.array([0,1,0,0,0,0])
    
                
            feature1=tf.concat([feature1,pressure1],axis=0)
            feature2=tf.concat([feature2,pressure2],axis=0)
            feature3=tf.concat([feature3,flowrate1],axis=0)
            feature4=tf.concat([feature4,flowrate2],axis=0)
            feature5=tf.concat([feature5,age],axis=0)
            target_dataset=tf.concat([target_dataset,target],axis=0)
            
        name1="..\samples\CoW_"+str(j)+"_00.csv"    
        path="..\samples\CoW_"+str(j)+"_ICAL2"+str(i)+"0.csv"
        if os.path.isfile(path) and os.path.isfile(name1):
            name1="..\samples\CoW_"+str(j)+"_ICAL200.csv"
            vessel2 = np.loadtxt(name1,delimiter=',')
            name1="..\samples\CoW_"+str(j)+"_00.csv"
            vessel1 = np.loadtxt(name1,delimiter=',')
            
            if j==20 or j ==21 or j ==22 or j==23:
                pressure1=np.square(np.delete(vessel1, [0,1,2,3,5], 1)/beta+np.sqrt(A0))                
                flowrate1=np.delete(vessel1, [0,1,3,4,5], 1)
                
            else:
                pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                flowrate1=np.delete(vessel1, [0,2,3], 1)
            
            pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
            flowrate2=np.delete(vessel2, [0,2,3], 1)
            
            
            age=np.tile(age1,(len(flowrate1),1))
        
            
            target=np.array([0,1,0,0,0,0])
    
                
            feature1=tf.concat([feature1,pressure1],axis=0)
            feature2=tf.concat([feature2,pressure2],axis=0)
            feature3=tf.concat([feature3,flowrate1],axis=0)
            feature4=tf.concat([feature4,flowrate2],axis=0)
            feature5=tf.concat([feature5,age],axis=0)
            target_dataset=tf.concat([target_dataset,target],axis=0)

        name1="..\samples\CoW_"+str(j)+"_01.csv"
        path="..\samples\CoW_"+str(j)+"_ICAR"+str(i)+"0.csv"
        if os.path.isfile(path) and os.path.isfile(name1):
            name1="..\samples\CoW_"+str(j)+"_ICAR01.csv"
            vessel2 = np.loadtxt(name1,delimiter=',')
            name1="..\samples\CoW_"+str(j)+"_01.csv"
            vessel1 = np.loadtxt(name1,delimiter=',')
            
            if j==20 or j ==21 or j ==22 or j==23:
                pressure1=np.square(np.delete(vessel1, [0,1,2,3,5], 1)/beta+np.sqrt(A0))                
                flowrate1=np.delete(vessel1, [0,1,3,4,5], 1)
                
            else:
                pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                flowrate1=np.delete(vessel1, [0,2,3], 1)
            
            pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
            flowrate2=np.delete(vessel2, [0,2,3], 1)
            
            
            age=np.tile(age1,(len(flowrate1),1))
        
            
            target=np.array([0,1,0,0,0,0])
    
                
            feature1=tf.concat([feature1,pressure1],axis=0)
            feature2=tf.concat([feature2,pressure2],axis=0)
            feature3=tf.concat([feature3,flowrate1],axis=0)
            feature4=tf.concat([feature4,flowrate2],axis=0)
            feature5=tf.concat([feature5,age],axis=0)
            target_dataset=tf.concat([target_dataset,target],axis=0)
            
        name1="..\samples\CoW_"+str(j)+"_01.csv"
        path="..\samples\CoW_"+str(j)+"_ICAR2"+str(i)+"0.csv"
        if os.path.isfile(path) and os.path.isfile(name1):
            name1="..\samples\CoW_"+str(j)+"_ICAR201.csv"
            vessel2 = np.loadtxt(name1,delimiter=',')
            name1="..\samples\CoW_"+str(j)+"_01.csv"
            vessel1 = np.loadtxt(name1,delimiter=',')
            
            if j==20 or j ==21 or j ==22 or j==23:
                pressure1=np.square(np.delete(vessel1, [0,1,2,3,5], 1)/beta+np.sqrt(A0))                
                flowrate1=np.delete(vessel1, [0,1,3,4,5], 1)
                
            else:
                pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                flowrate1=np.delete(vessel1, [0,2,3], 1)
            
            pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
            flowrate2=np.delete(vessel2, [0,2,3], 1)
            
            
            age=np.tile(age1,(len(flowrate1),1))
        
            
            target=np.array([0,1,0,0,0,0])
    
                
            feature1=tf.concat([feature1,pressure1],axis=0)
            feature2=tf.concat([feature2,pressure2],axis=0)
            feature3=tf.concat([feature3,flowrate1],axis=0)
            feature4=tf.concat([feature4,flowrate2],axis=0)
            feature5=tf.concat([feature5,age],axis=0)
            target_dataset=tf.concat([target_dataset,target],axis=0)
            
        name1="..\samples\CoW_"+str(j)+"_00.csv"
        path="..\samples\CoW_"+str(j)+"_PCAL"+str(i)+"0.csv"
        if os.path.isfile(path) and os.path.isfile(name1):
            name1="..\samples\CoW_"+str(j)+"_PCAL00.csv"
            vessel2 = np.loadtxt(name1,delimiter=',')
            name1="..\samples\CoW_"+str(j)+"_00.csv"
            vessel1 = np.loadtxt(name1,delimiter=',')
            
            if j==20 or j ==21 or j ==22 or j==23:
                pressure1=np.square(np.delete(vessel1, [0,1,2,3,5], 1)/beta+np.sqrt(A0))                
                flowrate1=np.delete(vessel1, [0,1,3,4,5], 1)
                
            else:
                pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                flowrate1=np.delete(vessel1, [0,2,3], 1)
            
            pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
            flowrate2=np.delete(vessel2, [0,2,3], 1)
            
            
            age=np.tile(age1,(len(flowrate1),1))
        
            
            target=np.array([0,0,1,0,0,0])
    

            feature1=tf.concat([feature1,pressure1],axis=0)
            feature2=tf.concat([feature2,pressure2],axis=0)
            feature3=tf.concat([feature3,flowrate1],axis=0)
            feature4=tf.concat([feature4,flowrate2],axis=0)
            feature5=tf.concat([feature5,age],axis=0)
            target_dataset=tf.concat([target_dataset,target],axis=0)
        
        name1="..\samples\CoW_"+str(j)+"_00.csv"
        path="..\samples\CoW_"+str(j)+"_PCAL2"+str(i)+"0.csv"
        if os.path.isfile(path) and os.path.isfile(name1):
            name1="..\samples\CoW_"+str(j)+"_PCAL200.csv"
            vessel2 = np.loadtxt(name1,delimiter=',')
            name1="..\samples\CoW_"+str(j)+"_00.csv"
            vessel1 = np.loadtxt(name1,delimiter=',')
            
            if j==20 or j ==21 or j ==22 or j==23:
                pressure1=np.square(np.delete(vessel1, [0,1,2,3,5], 1)/beta+np.sqrt(A0))                
                flowrate1=np.delete(vessel1, [0,1,3,4,5], 1)
                
            else:
                pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                flowrate1=np.delete(vessel1, [0,2,3], 1)
            
            pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
            flowrate2=np.delete(vessel2, [0,2,3], 1)
            
            
            age=np.tile(age1,(len(flowrate1),1))
        
            
            target=np.array([0,0,0,1,0,0])
    
                
            feature1=tf.concat([feature1,pressure1],axis=0)
            feature2=tf.concat([feature2,pressure2],axis=0)
            feature3=tf.concat([feature3,flowrate1],axis=0)
            feature4=tf.concat([feature4,flowrate2],axis=0)
            feature5=tf.concat([feature5,age],axis=0)
            target_dataset=tf.concat([target_dataset,target],axis=0)
        
        name1="..\samples\CoW_"+str(j)+"_01.csv"
        path="..\samples\CoW_"+str(j)+"_PCAR"+str(i)+"0.csv"
        if os.path.isfile(path) and os.path.isfile(name1):
            name1="..\samples\CoW_"+str(j)+"_PCAR01.csv"
            vessel2 = np.loadtxt(name1,delimiter=',')
            name1="..\samples\CoW_"+str(j)+"_01.csv"
            vessel1 = np.loadtxt(name1,delimiter=',')
            
            if j==20 or j ==21 or j ==22 or j==23:
                pressure1=np.square(np.delete(vessel1, [0,1,2,3,5], 1)/beta+np.sqrt(A0))                
                flowrate1=np.delete(vessel1, [0,1,3,4,5], 1)
                
            else:
                pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                flowrate1=np.delete(vessel1, [0,2,3], 1)
            
            pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
            flowrate2=np.delete(vessel2, [0,2,3], 1)
            
            
            age=np.tile(age1,(len(flowrate1),1))
        
            
            target=np.array([0,0,1,0,0,0])
    
                
            feature1=tf.concat([feature1,pressure1],axis=0)
            feature2=tf.concat([feature2,pressure2],axis=0)
            feature3=tf.concat([feature3,flowrate1],axis=0)
            feature4=tf.concat([feature4,flowrate2],axis=0)
            feature5=tf.concat([feature5,age],axis=0)
            target_dataset=tf.concat([target_dataset,target],axis=0)
        
        name1="..\samples\CoW_"+str(j)+"_01.csv"
        path="..\samples\CoW_"+str(j)+"_PCAR2"+str(i)+"0.csv"
        if os.path.isfile(path) and os.path.isfile(name1):
            name1="..\samples\CoW_"+str(j)+"_PCAR200.csv"
            vessel2 = np.loadtxt(name1,delimiter=',')
            name1="..\samples\CoW_"+str(j)+"_01.csv"
            vessel1 = np.loadtxt(name1,delimiter=',')
            
            if j==20 or j ==21 or j ==22 or j==23:
                pressure1=np.square(np.delete(vessel1, [0,1,2,3,5], 1)/beta+np.sqrt(A0))                
                flowrate1=np.delete(vessel1, [0,1,3,4,5], 1)
                
            else:
                pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                flowrate1=np.delete(vessel1, [0,2,3], 1)
            
            pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
            flowrate2=np.delete(vessel2, [0,2,3], 1)
            
            
            age=np.tile(age1,(len(flowrate1),1))
        
            
            target=np.array([0,0,0,1,0,0])
    
                
            feature1=tf.concat([feature1,pressure1],axis=0)
            feature2=tf.concat([feature2,pressure2],axis=0)
            feature3=tf.concat([feature3,flowrate1],axis=0)
            feature4=tf.concat([feature4,flowrate2],axis=0)
            feature5=tf.concat([feature5,age],axis=0)
            target_dataset=tf.concat([target_dataset,target],axis=0)
        
        name1="..\samples\CoW_"+str(j)+"_00.csv"
        path="..\samples\CoW_"+str(j)+"_VAL"+str(i)+"0.csv"
        if os.path.isfile(path) and os.path.isfile(name1):
            name1="..\samples\CoW_"+str(j)+"_VAL00.csv"
            vessel2 = np.loadtxt(name1,delimiter=',')
            name1="..\samples\CoW_"+str(j)+"_00.csv"
            vessel1 = np.loadtxt(name1,delimiter=',')
            
            if j==20 or j ==21 or j ==22 or j==23:
                pressure1=np.square(np.delete(vessel1, [0,1,2,3,5], 1)/beta+np.sqrt(A0))                
                flowrate1=np.delete(vessel1, [0,1,3,4,5], 1)
                
            else:
                pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                flowrate1=np.delete(vessel1, [0,2,3], 1)
            
            pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
            flowrate2=np.delete(vessel2, [0,2,3], 1)
            
            
            age=np.tile(age1,(len(flowrate1),1))
        
            
            target=np.array([0,0,0,0,1,0])
    
                
            feature1=tf.concat([feature1,pressure1],axis=0)
            feature2=tf.concat([feature2,pressure2],axis=0)
            feature3=tf.concat([feature3,flowrate1],axis=0)
            feature4=tf.concat([feature4,flowrate2],axis=0)
            feature5=tf.concat([feature5,age],axis=0)
            target_dataset=tf.concat([target_dataset,target],axis=0)
        
        name1="..\samples\CoW_"+str(j)+"_01.csv"
        path="..\samples\CoW_"+str(j)+"_VAR"+str(i)+"1.csv"
        if os.path.isfile(path) and os.path.isfile(name1):
            name1="..\samples\CoW_"+str(j)+"_VAR01.csv"
            vessel2 = np.loadtxt(name1,delimiter=',')
            name1="..\samples\CoW_"+str(j)+"_01.csv"
            vessel1 = np.loadtxt(name1,delimiter=',')
            
            if j==20 or j ==21 or j ==22 or j==23:
                pressure1=np.square(np.delete(vessel1, [0,1,2,3,5], 1)/beta+np.sqrt(A0))                
                flowrate1=np.delete(vessel1, [0,1,3,4,5], 1)
                
            else:
                pressure1=np.square(np.delete(vessel1, [0,1,3], 1)/beta+np.sqrt(A0))
                flowrate1=np.delete(vessel1, [0,2,3], 1)
            
            pressure2=np.square(np.delete(vessel2, [0,1,3], 1)/beta+np.sqrt(A0))
            flowrate2=np.delete(vessel2, [0,2,3], 1)
            
            
            age=np.tile(age1,(len(flowrate1),1))
        
            
            target=np.array([0,0,0,0,1,0])
    
                
            feature1=tf.concat([feature1,pressure1],axis=0)
            feature2=tf.concat([feature2,pressure2],axis=0)
            feature3=tf.concat([feature3,flowrate1],axis=0)
            feature4=tf.concat([feature4,flowrate2],axis=0)
            feature5=tf.concat([feature5,age],axis=0)
            target_dataset=tf.concat([target_dataset,target],axis=0)
        
        i=68
        name1="..\samples\CoW_"+str(j)+"_00.csv"
        name2="..\samples\CoW_"+str(j)+"_MCAL"+str(int(i))+"0.csv"
        if os.path.isfile(name2) and os.path.isfile(name1):
            vessel1 = np.loadtxt(name1,delimiter=',')
            vessel2 = np.loadtxt(name2,delimiter=',')
            
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
            
            target=np.array([0,0,0,0,0,1])
    

            feature1=tf.concat([feature1,pressure1],axis=0)
            feature2=tf.concat([feature2,pressure2],axis=0)
            feature3=tf.concat([feature3,flowrate1],axis=0)
            feature4=tf.concat([feature4,flowrate2],axis=0)
            feature5=tf.concat([feature5,age],axis=0)
            target_dataset=tf.concat([target_dataset,target],axis=0)
            
        i=71
        name1="..\samples\CoW_"+str(j)+"_01.csv"
        name2="..\samples\CoW_"+str(j)+"_MCAR"+str(int(i))+"1.csv"
        if os.path.isfile(name2) and os.path.isfile(name1):
            vessel1 = np.loadtxt(name1,delimiter=',')
            vessel2 = np.loadtxt(name2,delimiter=',')
            
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
            
            target=np.array([0,0,0,0,0,1])
    

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
    
    
    if mode==0:
        data=np.hstack([feature1,feature2,feature3,feature4])
    elif mode==1:
        data=np.hstack([feature3,feature4])
    elif mode==2:
        data=np.hstack([feature1,feature2])
    target_dataset=np.hstack([target_dataset])
    target_dataset=np.hstack([target_dataset])
    #### output data #####
    
    y_dataset1 = []
    x_dataset1 = []
    counter=0
    counterr=0
    N_totat=int(len(feature1)/N_t)
    for j in range(0,N_totat):
        x_dataset1.append(data[N_t*counter:N_t*(counter+1),:])
        y_dataset1.append(target_dataset[counterr:counterr+6])
        counter=counter+1
        counterr=counterr+6
            
            
    y_dataset= np.array(y_dataset1)
    x_dataset= np.array(x_dataset1)
    
    test_size=len(x_dataset)-train_size
    train_size=train_size
    
    indices = np.random.permutation(x_dataset.shape[0])
    training_idx, test_idx = indices[:int(train_size)], indices[int(train_size):]
    x_train, x_test = x_dataset[training_idx,:], x_dataset[test_idx,:]
    y_train, y_test = y_dataset[training_idx,:], y_dataset[test_idx,:]

    input_layer = keras.layers.Input(shape=(N_t,n_features))
    layer1=tf.keras.layers.Conv1D(25, kernel_size=5, strides=1, padding='valid', activation='tanh')(input_layer)
    layer2=tf.keras.layers.Conv1D(10, kernel_size=5, strides=1, padding='valid', activation='tanh')(layer1)
    pool1=tf.keras.layers.MaxPool1D(pool_size=2)(layer2)
    layer3=tf.keras.layers.Conv1D(5, kernel_size=5, strides=1, padding='valid', activation='tanh')(pool1)
    layer4=tf.keras.layers.Conv1D(2, kernel_size=5, strides=1, padding='valid', activation='tanh')(layer3)
    pool2=tf.keras.layers.MaxPool1D(pool_size=2)(layer4)
    flat=tf.keras.layers.Flatten()(pool2)
    dense1=tf.keras.layers.Dense(100, activation='tanh')(flat)
    dense2=tf.keras.layers.Dense(50, activation='tanh')(dense1)
    dense3=tf.keras.layers.Dense(25, activation='tanh')(dense2)
    dense4=tf.keras.layers.Dense(10, activation='tanh')(dense3)
    dense5=tf.keras.layers.Dense(10, activation='tanh')(dense4)
    #input_layer2=keras.layers.Input(shape=(1,))
    #concentrate=tf.keras.layers.Concatenate()([dense5,input_layer2])
    dense6=tf.keras.layers.Dense(10, activation='tanh')(dense5)
    output=tf.keras.layers.Dense(6, activation='softmax')(dense6)
    #output=tf.keras.layers.Dense(2, activation='sigmoid')(dense5)
    model = tf.keras.models.Model(inputs=[input_layer], outputs=output)
    

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    
    
    model.compile(loss='BinaryCrossentropy', metrics=['accuracy'], optimizer='adam')
    model.fit([x_train], y_train, batch_size=16, epochs=10000)
    name='model_segment_LVO_full_'+str(train_size)+'.h5'
    model.save(name)
    
    A=model.predict([x_train])
    predict=np.around(A)
    
    pred=np.zeros(train_size)
    true=np.ones(train_size)
    
    for i in range(0,train_size):
        if np.array_equal(predict[i,:],y_train[i,:]):
           pred[i]=1   
    
    
    F1_train=f1_score(true, pred, average='micro', zero_division=0)
    
    
    A=model.predict([x_test])
    predict=np.around(A)
    
    pred=np.zeros(int(test_size))
    true=np.ones(int(test_size))
    
    for i in range(0,int(test_size)):
        if np.array_equal(predict[i,:],y_test[i,:]):
           pred[i]=1  
    
    F1_test=f1_score(y_test, predict, average='micro', zero_division=0)
    
    
        
    
    cma=np.zeros((6,6))
    for i in range(0,len(A)):
        for j in range(0,6):
            for k in range(0,6):
                if predict[i,j]==1 and y_test[i,k]==1:
                    cma[k,j]=cma[k,j]+1
                    
    for i in range(0,6):
        number=np.sum(cma[i,:])
        cma[i,:]=cma[i,:]/number
                
    cma=np.round(cma,decimals=2)
    print(F1_train,F1_test)

    

    fpr, tpr,_ = metrics.roc_curve(y_test[:,i], A[:,i],pos_label=1,drop_intermediate=False)
    name='fpr_LVO_full_'+str(train_size)+'_'
    np.save(name,fpr)
    
    name='tpr_LVO_full_'+str(train_size)+'_'
    np.save(name,tpr)
    
    name='cma_LVO_full_'+str(train_size)+'_'
    np.save(name,cma)
    
    name='f1test_LVO_full_'+str(train_size)+'_'
    np.save(name,F1_test)
    
    name='P_nor_LVO_full_'+str(train_size)+'_'
    np.save(name,A_mean)
    
    name='P_std_LVO_full_'+str(train_size)+'_'
    np.save(name,A_std)
    
    name='V_nor_LVO_full_'+str(train_size)+'_'
    np.save(name,V_mean)
    
    name='V_std_LVO_full_'+str(train_size)+'_'
    np.save(name,V_std)
    
    name='Age_nor_LVO_full_'+str(train_size)+'_'
    np.save(name,Age_mean)
    
    name='Age_std_LVO_full_'+str(train_size)+'_'
    np.save(name,Age_std)
        

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
    
    
plot_confusion_matrix(cm           = cma, 
                  normalize    = False,
                  target_names = ['MVO', 'SVO', 'SVO', 'SVO', 'SVO', 'SVO'],
                  title        = "")
    
