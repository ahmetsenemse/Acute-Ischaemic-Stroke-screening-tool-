import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import pickle
import os
import artery_area as ata
import tensorflow as tf


N_mu=1000
numbermodes=1
N_h=numbermodes*1
N_param = 11

train_size=int(4*N_mu/5)

indices = np.random.permutation(N_mu)
training_idx, test_idx = indices[:int(train_size)], indices[int(train_size):]
empty=[]
train_size=0
for i in training_idx:
    name="..\samples/CoW_"+str(i)+"_0"+str(0)+".csv"
    name1="..\samples/CoW_"+str(i)+"_0"+str(1)+".csv"
    if os.path.isfile(name) and os.path.isfile(name1):
        train_size=train_size+1
    else:
        empty.append(i)
    
name="..\samples/CoW_0_00.csv"
s = np.loadtxt(name,delimiter=',')
v_dt=np.delete(s,[1,2,3],1)
N_t=len(v_dt)
N_s=train_size
N_h=numbermodes*N_t

t_mu = np.zeros((N_param,N_s))
S = np.zeros((N_h,N_s))


samples=np.zeros((N_t,1,numbermodes))



counter=0
for i in training_idx:
    name="..\samples/CoW_"+str(i)+"_0"+str(0)+".csv"
    name1="..\samples/CoW_"+str(i)+"_0"+str(1)+".csv"
    if os.path.isfile(name) and os.path.isfile(name1):
    
        v5 = np.loadtxt(name,delimiter=',')
        v6 = np.loadtxt(name1,delimiter=',')
        
        name="..\samples/CoW_"+str(i)+"_param.csv" #INCLUDE Branchial information +HR
        param = np.loadtxt(name,delimiter=',')[0,:]
        
        name='..\samples\Age'+str(i)+'.npy'
        age1=np.load(name)
        
        param=np.hstack([param,age1])
        
        

        v5 = np.delete(v5,[3],1)
        v6 = np.delete(v6,[3],1)
    
        samples[:,:,0]=v5[:,2:3]
        #samples[:,:,2]=v5[:,2:3]
#        samples[:,:,1]=v6[:,2:3]
        #samples[:,:,4]=v6[:,2:3]
        S[0:200,counter]=samples[:,0,0]
        #S[200:400,counter]=samples[:,0,1]
        t_mu[:,counter]=param[:]
        counter=counter+1
                

def sdv(S):    
    U, s, Z = np.linalg.svd(S, full_matrices=False)
    
    return U, s, Z

U, s, Z=sdv(S)

s = np.diag(s)


eps_tol = 10e-4

error_num = 0;
error_den = 0;

for i in range(0,min(N_h,N_s)):
    error_den = error_den + s[i,i]
    
error_comp = error_num/error_den;

L = 0;
while (error_comp<=(1-eps_tol)):
   error_num = error_num + s[L,L];
   error_comp = error_num/error_den;
   L = L+1;

    
V = U[:,0:L]

Storage_matrix=np.dot(np.transpose(V),S)

mumj =np.transpose( t_mu)

kernel = 1.0 * RBF()
clf_param=[]

for i in range(0,L):
    clf_param.append(GaussianProcessRegressor(kernel=kernel,random_state=100,n_restarts_optimizer=6))


output=np.zeros((train_size,L))
for l in range(0,L): 
    phi_var_k = Storage_matrix[l,:]
    clf_param[l].fit(mumj, phi_var_k)  
    output[:,l]=phi_var_k
        
        
        
#normalization

mumjmax=np.max(mumj,axis=0)
mumjmin=np.min(mumj,axis=0)
mumjnor=(mumj-mumjmin)/(mumjmax-mumjmin)


outputmax=np.max(output,axis=0)
outputmin=np.min(output,axis=0)
outputnor=(output-outputmin)/(outputmax-outputmin)

initializer='GlorotUniform'
activation='relu'
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(10, activation=activation, kernel_initializer=initializer))
model.add(tf.keras.layers.Dense(10, activation=activation, kernel_initializer=initializer))
model.add(tf.keras.layers.Dense(10, activation=activation, kernel_initializer=initializer))
model.add(tf.keras.layers.Dense(10, activation=activation, kernel_initializer=initializer))
model.add(tf.keras.layers.Dense(10, activation=activation, kernel_initializer=initializer))
model.add(tf.keras.layers.Dense(10, activation=activation, kernel_initializer=initializer))
model.add(tf.keras.layers.Dense(10, activation=activation, kernel_initializer=initializer))
model.add(tf.keras.layers.Dense(10, activation=activation, kernel_initializer=initializer))
model.add(tf.keras.layers.Dense(10, activation=activation, kernel_initializer=initializer))
model.add(tf.keras.layers.Dense(output.shape[1]))


class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('loss')<1e-12:
            self.model.stop_training = True


model.compile(optimizer='adam', loss='mse')
model.fit(mumjnor, output, batch_size=32, epochs=100000000,callbacks=[CustomCallback()])
        
        
    
i=training_idx[0]
for j in range(0,1):
    name="..\samples/CoW_"+str(i)+"_param.csv"
    param = np.loadtxt(name,delimiter=',')[0,:]
    
    name='..\samples\Age'+str(i)+'.npy'
    age1=np.load(name)
    
    param=np.hstack([param,age1])
    mu=param
    
    munor=(mu-mumjmin)/(mumjmax-mumjmin)
    Prediction=model.predict(np.reshape(munor,(1,11)))#*(outputmax-outputmin)+outputmin
    
    qhat=np.zeros((1,L))
    qhat1=np.zeros((1,L))
    counter=0
    for l in range(0,L): 
        phi_var_k_mu = clf_param[l].predict(np.reshape(mu, (1, N_param)))
        phi_var_k_mu1 = Prediction[0,l]
        
        qhat[:,l] = phi_var_k_mu;
        qhat1[:,l] = phi_var_k_mu1;
        
    if j==0:        
        v_urb =np.matmul(V,np.transpose(qhat))
        v_urb1 =np.matmul(V,np.transpose(qhat1))

    elif j==1:        
        v_urb1 =np.matmul(V,np.transpose(qhat))
    elif j==2:        
        v_urb2 =np.matmul(V,np.transpose(qhat))
    elif j==3:        
        v_urb3 =np.matmul(V,np.transpose(qhat))        
    elif j==4:        
        v_urb4 =np.matmul(V,np.transpose(qhat))
        
with open('clf_param5.pkl','wb') as f:
    pickle.dump(clf_param,f)
np.save("L5",L)
np.save("V5",V)

S = np.zeros((N_h,1))
samplestest=np.zeros((N_t,1,numbermodes))
name="..\samples/CoW_"+str(i)+"_0"+str(0)+".csv"
v5 = np.loadtxt(name,delimiter=',')
name="..\samples/CoW_"+str(i)+"_0"+str(1)+".csv"
v6 = np.loadtxt(name,delimiter=',')

a = np.ones(1)*0.2802 #
b =  np.ones(1)*(-0.5053)*1000 #m-1
c =  np.ones(1)*0.1324 #
d =  np.ones(1)*(-0.01114)*1000 #m-1
E=0.8*1e6

A0=ata.CCA(age1)
r=np.sqrt(A0/np.pi)
h=r*(a*np.exp(b*r)+c*np.exp(d*r))
beta=(4/3)*((np.sqrt(np.pi)*E*h)/(A0))

v5 = np.delete(v5,[3],1)
samplestest[:,:,0]=v5[:,2:3]
#samplestest[:,:,1]=v5[:,2:3]
#v6 = np.delete(v6,[3],1)
#samplestest[:,:,1]=v6[:,2:3]
#samplestest[:,:,3]=v6[:,2:3]

S[0:200,0]=samplestest[:,0,0]
#S[200:400,0]=samplestest[:,0,1]

    
    
#error=np.square(S[1,:]-v_urb[1,:])
#error1=np.square(S[1,:]-v_urb1[1,:])
#error2=np.square(S[1,:]-v_urb2[1,:])
#error3=np.square(S[1,:]-v_urb3[1,:])
#error4=np.square(S[1,:]-v_urb4[1,:])

colors = plt.cm.Greys(np.linspace(0.3, 1, 5))

fig1 = plt.figure(1,figsize=(12, 6), dpi=800, facecolor='w', frameon = False)
ax11 = fig1.add_subplot(111)
#ax12 = fig1.add_subplot(222)
#ax13 = fig1.add_subplot(223)
#ax14 = fig1.add_subplot(224)
plt.rcParams.update({'font.size': 20})


ax11.plot(v_dt-40,S[0:200,0],'r',linewidth=1, markersize=0.5, label='%0 noise')
ax11.plot(v_dt-40,v_urb[0:200,0],'b',linewidth=1, markersize=0.5, label='%5 noise')
ax11.plot(v_dt-40,v_urb1[0:200,0],'g',linewidth=1, markersize=0.5, label='%5 noise')


#ax11.plot(v_dt-40,S[0:200,0],color=colors[4,:],linewidth=1, markersize=0.5, label='%0 noise')
#ax11.plot(v_dt-40,v_urb[0:200,0],color=colors[3,:],linewidth=1, markersize=0.5, label='%5 noise')
#ax11.plot(v_dt-40,error2,color=colors[2,:],linewidth=1, markersize=0.5, label='%10 noise')
#ax11.plot(v_dt-40,error3,color=colors[1,:],linewidth=1, markersize=0.5, label='%15 noise')
#ax11.plot(v_dt-40,error4,color=colors[0,:],linewidth=1, markersize=0.5, label='%20 noise')

#ax12.plot(v_dt-40,v_urb[2,:]*2,'b-',linewidth=1, markersize=0.5, label='Predicted velocity Vessel1')
#ax12.plot(v_dt-40,S[2,:]*2,'r-',linewidth=1, markersize=0.5, label='Reference velocity Vessel1')

#ax13.plot(v_dt-40,beta*(0.5*np.sqrt(v_urb[1,:])-np.sqrt(A0))*0.0075+320,'b-',linewidth=1, markersize=0.5, label='Predicted velocity Vessel1')
#ax13.plot(v_dt-40,beta*(0.5*np.sqrt(S[1,:])-np.sqrt(A0))*0.0075+320,'r-',linewidth=1, markersize=0.5, label='Reference velocity Vessel1')

#ax14.plot(v_dt-40,beta*(0.5*np.sqrt(v_urb[3,:])-np.sqrt(A0))*0.0075+320,'b-',linewidth=1, markersize=0.5, label='Predicted velocity Vessel1')
#ax14.plot(v_dt-40,beta*(0.5*np.sqrt(S[3,:])-np.sqrt(A0))*0.0075+320,'r-',linewidth=1, markersize=0.5, label='Reference velocity Vessel1')

#ax11.set_xlabel('Time [s]', fontsize=24)
#ax11.set_ylabel('Velocity [m/s]', fontsize=24)
ax11.grid(visible=bool)
#ax11.legend(fontsize=14)

#ax12.set_xlabel('Time [s]', fontsize=24)
#ax12.set_ylabel('Velocity [m/s]', fontsize=24)
#ax12.grid(visible=bool)

#ax13.set_xlabel('Time [s]', fontsize=24)
#ax13.set_ylabel('Pressure [mmHg]', fontsize=24)
#ax13.grid(visible=bool)

#ax14.set_xlabel('Time [s]', fontsize=24)
#ax14.set_ylabel('Pressure [mmHg]', fontsize=24)
#ax14.grid(visible=bool)
