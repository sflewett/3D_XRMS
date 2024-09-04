# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:00:29 2024

@author: sflewett
"""

#Converting some old MATLAB code to Python in order to generate Mx,My,Mz arrays
import numpy as np
from scipy.ndimage import distance_transform_edt
delta_x=5e-9
Pi=3.141592654; 
n=512
dww=3.0
period_parameter=5
nz=20
crossover=16
z=np.linspace(0,nz-1,nz)
dw_orientation=(np.tanh((z-crossover)/2)+1)*np.pi/2
x_bias=0.0
stripe_worm=0
disorder_exponent=2.2
X,Y=np.meshgrid(np.linspace(-n/2,n/2-1,n),np.linspace(-n/2,n/2-1,n))
#defining the X Y and Z co-ordinates of the magnetic sample
R=np.sqrt(X**2+Y**2)
phi1=np.arctan2(Y,X)
#Generating the magnetic domains to use as a test case
#real part
random_array = np.random.uniform(-1,1,(n,n))

distribution=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(random_array)))


domain_period_delta_x=2*Pi
domain_freq_pixels=n/(domain_period_delta_x*period_parameter)

freq_filt=1/(np.abs(((R-domain_freq_pixels)/np.max(R)))**disorder_exponent+0.000002)
stripe=np.sin(phi1)**stripe_worm
in_arr_FT=distribution*freq_filt*stripe


domains1=np.real(np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(in_arr_FT))));
domains_real=domains1/np.max(domains1);
a=domains_real
b=a.copy();


b[a>0]=1;b[a<0]=-1;
for j in range(n):
    if np.abs(np.sum(a[:,j]/n))<1e-6:
        b[:,j]=b[:,j+1];
 
surface_pattern_binary=b;

F = np.gradient(a);
FX=F[0]
FY=F[1]
F = np.gradient(b);
FX2=F[0]
FY2=F[1]

c=(np.abs(FX2)+np.abs(FY2));
d=c;
d[c<0.1]=0;
d[c>0.1]=1;
distance,indices= distance_transform_edt(d==0,return_indices=True)
Grad_x=FX/np.sqrt(FX**2+FY**2);Grad_y=FY/np.sqrt(FX**2+FY**2);
indices_y=indices[0,:,:];
indices_x=indices[1,:,:];
indices_x2=indices_x;
indices_y2=indices_y;
indices_x[indices_x2>n]=n;indices_y[indices_y2>n]=n;
#%%distance from the domain walls"%%
Grad_x2=Grad_x-Grad_x
Grad_y2=Grad_x2.copy()

for jj in range(n):
    for kk in range(n):
        Grad_x2[jj,kk]=Grad_x[indices_y[jj,kk],indices_x[jj,kk]];
        Grad_y2[jj,kk]=Grad_y[indices_y[jj,kk],indices_x[jj,kk]];


Grad_x=Grad_x2/np.sqrt(Grad_x2**2+Grad_y2**2);Grad_y=Grad_y2/np.sqrt(Grad_x2**2+Grad_y2**2);


#%normalized gradient
#%Defining parameters 
X_rot=np.zeros((n,n));
Y_rot=np.zeros((n,n));
Z_rot=np.zeros((n,n));
Mx=np.zeros((n,n));
My=np.zeros((n,n));
Mz=np.zeros((n,n));
test2=np.zeros((n,n));


n1=n;

phi_multiplier=[1];
delta_exponent=[1.5];
#parameters for the analytical input model


spb_interp=surface_pattern_binary
distance_interp=distance+0.5
Grad_x_interp=Grad_x
Grad_y_interp=Grad_y
Grad_x_interp2=Grad_x_interp/np.sqrt(Grad_x_interp**2+Grad_y_interp**2);
Grad_y_interp2=Grad_y_interp/np.sqrt(Grad_x_interp**2+Grad_y_interp**2);
Grad_x_interp=Grad_x_interp2;
Grad_y_interp=Grad_y_interp2;

average_z=np.tanh(distance/(dww));

output=np.zeros((n,n))
#Initializing the output vector


Mx=np.zeros((n,n,nz))
My=np.zeros((n,n,nz))
Mz=np.zeros((n,n,nz))


temp2=np.ones(len(z))
for j in range (len(z)):
    if j > crossover:
        temp2[j]=-1

for j in range(nz):
    bloch_neel=spb_interp*dw_orientation[j]
    Mz[:,:,j]=spb_interp*average_z
    My[:,:,j]=np.sqrt(1-Mz[:,:,j]**2)*(spb_interp*Grad_x_interp*np.sin(bloch_neel)+Grad_y_interp*np.cos(bloch_neel))
    Mx[:,:,j]=np.sqrt(1-Mz[:,:,j]**2)*(spb_interp*np.abs(Grad_y_interp)*np.sin(bloch_neel)+Grad_x_interp*np.cos(bloch_neel))

Mx=Mx+x_bias
My=My

My2=My/np.sqrt(Mx**2+My**2+Mz**2);
Mx2=Mx/np.sqrt(Mx**2+My**2+Mz**2);
Mz2=Mz/np.sqrt(Mx**2+My**2+Mz**2);

mx3=np.abs(np.ndarray.flatten(Mx2))

my3=np.ndarray.flatten(My2)

mz3=np.ndarray.flatten(Mz2)
np.savetxt("../Magnetization Files/Biased_worms_SRI_z.csv", mz3, delimiter=",")

np.savetxt("../Magnetization Files/Biased_worms_SRI_y.csv", my3, delimiter=",")

np.savetxt("../Magnetization Files/Biased_worms_SRI_x.csv", mx3, delimiter=",")