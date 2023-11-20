import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import pickle

N_mu=2
numbermodes=2
N_h=numbermodes*1
N_param = 11;
    
name="..\Snaps/CoW_0_0_5.csv"
s = np.loadtxt(name,delimiter=',')
v_dt=np.delete(s,[1,2,3,4,5],1)
N_t=len(v_dt)
N_s=N_t*N_mu

t_mu = np.zeros((N_param+1,N_s))
S = np.zeros((N_h,N_s))


snaps=np.zeros((N_t,1,numbermodes+1))

for i in range(0,N_mu):
    name="..\Snaps/CoW_"+str(i)+"_0_"+str(5)+".csv"
    v5 = np.loadtxt(name,delimiter=',')
    name="..\Snaps/CoW_"+str(i)+"_param.csv"
    param = np.loadtxt(name,delimiter=',')
    
    v5 = np.delete(v5,[1,3,5],1)
    snaps[:,:,0]=v5[:,0:1]
    snaps[:,:,1]=v5[:,1:2]
    snaps[:,:,2]=v5[:,2:3]

    for j in range(0,numbermodes):
        for k in range(0,N_t):
            S[j*1,k+i*N_t]=snaps[k,0,j+1]
            t_mu[0,k+i*N_t]=snaps[k,0,0]
            
    for j in range(1,N_param+1):
        for k in range(0,N_t):
            t_mu[j,k+i*N_t]=param[j-1]

def sdv(S):    
    U, s, Z = np.linalg.svd(S, full_matrices=True)
    
    return U, s, Z


U, s, Z=sdv(S)

s = np.diag(s)


eps_tol = 10e-8

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

Pl_aux_t = np.zeros((N_t,N_mu))
Pl_aux_mu = np.zeros((N_t,N_mu,N_param))


for i in range(0,N_mu):
    Pl_aux_t[:,i] = np.transpose(t_mu)[i*N_t:N_t+i*N_t,0]
    for j in range(0,N_t):
        name="..\Snaps/CoW_"+str(i)+"_param.csv"
        param = np.loadtxt(name,delimiter=',')
        for k in range(0,N_param):
            Pl_aux_mu[:,i,k] = param[k]
        
Dtr =np.matmul(np.transpose(V),S)
Pl = np.zeros((N_t,N_mu,L))

for l in range(0,L):
   for i in range(0,N_mu):
       Pl[:,i,l] = Dtr[l,i*N_t:N_t+i*N_t]
       
       
Nt_tr=len(Pl)
Nmu_tr=len(Pl[2])
 
Pl_U = np.zeros((Nt_tr,Nt_tr,L))
Pl_Sig = np.zeros((Nt_tr,Nmu_tr,L))
Pl_Z = np.zeros((Nmu_tr,Nmu_tr,L))

for l in range(0,L):
    Pl1 = Pl[:,:,l]
    U1, s1, Z1 = np.linalg.svd(Pl1, full_matrices=True)
    s1 = np.diag(s1)
    Pl_U[:,:,l] = U1
    for k in range(0,len(s1)):
        Pl_Sig[k,k,l] = s1[k,k]
    Pl_Z[:,:,l] = Z1
    
Ql = np.zeros((1,L))

for l in range(0,L):
    
    Sig = Pl_Sig[:,:,l]
    
    error_num = 0;
    error_den = 0;
    
    
    for i in range(0,min(Nmu_tr,Nt_tr)):
        error_den = error_den + Sig[i,i]
    
    error_comp = error_num/error_den
    
    Ql_l = 0
    
    while (error_comp<=(1-eps_tol)):
        error_num = error_num + Sig[Ql_l,Ql_l];
        error_comp = error_num/error_den;
        Ql_l = Ql_l+1;
        
        
    Ql[0,l] = Ql_l;
    
    
Ql_list = Ql;
U_list = Pl_U;
Z_list = Pl_Z;
tni =Pl_aux_t[:,0]
mumj = np.transpose(np.stack((Pl_aux_mu[0,:,0],Pl_aux_mu[0,:,1],
                              Pl_aux_mu[0,:,2],Pl_aux_mu[0,:,3],
                              Pl_aux_mu[0,:,4],Pl_aux_mu[0,:,5],
                              Pl_aux_mu[0,:,6],Pl_aux_mu[0,:,7],
                              Pl_aux_mu[0,:,8],Pl_aux_mu[0,:,9],
                              Pl_aux_mu[0,:,10],
                              ),axis=0))

phi_k_t_list={}
phi_k_mu_list={}
kernel = 1.0 * RBF()

clf_time=[]
clf_param=[]

for i in range(0,int(np.sum(Ql))):
    clf_time.append(GaussianProcessRegressor(kernel=kernel,random_state=100,n_restarts_optimizer=4))
    clf_param.append(GaussianProcessRegressor(kernel=kernel,random_state=100,n_restarts_optimizer=4))

Qlmax = np.max(Ql_list)
v_phi_k_t = np.zeros((int(Qlmax),len(v_dt),L))
v_phi_var_k_mu = np.zeros((int(Qlmax),len(v_dt),L))
v_lambda_k = np.zeros((int(Qlmax),len(v_dt),L))
qhat = np.zeros((Nt_tr,L))


counter=0
for l in range(0,L): 
    U = U_list[:,:,l]
    Z = np.transpose(Z_list[:,:,l])
    Ql = Ql_list[0,l]

    for k in range(0,int(Ql)):
        print('l = %d out of %d, Ql = %d out of %d'%(l+1,L,k+1,Ql))
        
        phi_k = U[:,k]  
        phi_var_k = Z[:,k]
        
        clf_time[counter].fit(np.reshape(tni, (Nt_tr, 1)), phi_k)
        clf_param[counter].fit(mumj, phi_var_k)
        
        counter=counter+1
        
        
    
i=0
name="..\Snaps/CoW_"+str(i)+"_param.csv"
param = np.loadtxt(name,delimiter=',')
mu=param
counter=0
for l in range(0,L): 
    Ql = Ql_list[0,l]
    Sig = Pl_Sig[:,:,l]
    U = U_list[:,:,l]
    Z = np.transpose(Z_list[:,:,l])
    ql = 0;
    
    
    for k in range(0,int(Ql)):
        print('l = %d out of %d, Ql = %d out of %d'%(l+1,L,k+1,Ql))
        
        lambda_k = Sig[k,k]
        phi_k = U[:,k]
        phi_var_k = Z[:,k]
        
        phi_k_t  = phi_k
        #phi_var_k_mu = clf_param[counter].predict(np.reshape(mu, (1, N_param)))
        phi_var_k_mu = phi_var_k[0]

        v_phi_k_t[k,:,l] = np.transpose(phi_k_t)
        v_phi_var_k_mu[k,:,l] = phi_var_k_mu
        v_lambda_k[k,:,l] = lambda_k
        
        ql = ql + lambda_k*phi_k_t*phi_var_k_mu
        counter=counter+1
        
    qhat[:,l] = ql;
    
        
v_urb =np.matmul(V,np.transpose(qhat))

with open('clf_time5.pkl','wb') as f:
    pickle.dump(clf_time,f)
with open('clf_param5.pkl','wb') as f:
    pickle.dump(clf_param,f)
np.save("Pl_Sig5",Pl_Sig)
np.save("Ql_list5",Ql_list)
np.save("L5",L)
np.save("V5",V)
S = np.zeros((N_h,N_t))


name="..\Snaps/CoW_"+str(i)+"_0_"+str(5)+".csv"
v5 = np.loadtxt(name,delimiter=',')

v5 = np.delete(v5,[1,3,5],1)
snaps[:,:,0]=v5[:,1:2]
snaps[:,:,1]=v5[:,2:3]


for k in range(0,N_t):
    S[0,k]=snaps[k,0,0]
    S[1,k]=snaps[k,0,1]

fig1 = plt.figure(1,figsize=(22, 22), dpi=300, facecolor='w', frameon = False)
ax11 = fig1.add_subplot(111)


ax11.plot(v_dt,v_urb[1,:],'b-',linewidth=1, markersize=0.5, label='Predicted velocity Vessel1')
ax11.plot(v_dt,S[1,:],'r-',linewidth=1, markersize=0.5, label='Reference velocity Vessel1')

name="v"+str(20)
fig1.savefig(name)
