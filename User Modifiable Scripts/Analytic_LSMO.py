# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 09:09:18 2024

@author: sflewett
"""
import numpy as np

size_y=128
size_x=2
size_z=80
z=np.linspace(0,size_z-1,size_z)/4

dww="SS"#0.4*np.ones(len(z))#-0.00001*np.exp(-z)
dw_orientation=(np.tanh((z-2.0)))
spiral_type=11-10*np.exp(-z)
print(spiral_type)
skewness=1.0
#for angle_dw in np.linspace(np.pi/4,3*np.pi/8,6):
    #dw_orientation=angle_dw*np.ones(len(z))#(np.tanh((z+0)/1.6)+1)*np.pi/2
y=np.zeros(size_y)
''' for j in range(len(y)):
    jj=j%(size_y)
    if j ==0:
        y[j]=0
    if jj<size_y/2*skewness and j>0:
        dy=1/skewness
        y[j]=y[j-1]+dy
    if jj>=size_y*skewness and j>0:
        dy=size_y*2/(size_y-size_y/2*skewness)
        y[j]=y[j-1]+dy '''
y=np.linspace(-size_y/2,size_y/2-1,size_y)

#y=np.linspace(0,size_y-1,size_y)
y=y/size_y*2*np.pi
#print(y)
x_bias=0.0
mx=np.zeros((size_x,size_y,size_z))
my=np.zeros((size_x,size_y,size_z))
mz=np.zeros((size_x,size_y,size_z))
m_inplane=np.zeros((size_x,size_y,size_z))
orientation_x=np.zeros((size_x,size_y,size_z))
orientation_y=np.zeros((size_x,size_y,size_z))
if dww!="SS":
    for j in range(size_z):
        mz[:,:,j]=np.abs(np.tanh((np.sin(y))/dww[j]))*np.sign(np.sin(y))
        mx[:,:,j]=np.abs((1-np.tanh((np.sin(y))/dww[j])**2)**0.5*np.sin(dw_orientation[j])*np.sign(np.cos(y)))+x_bias
        my[:,:,j]=(1-np.tanh((np.sin(y))/dww[j])**2)**0.5*np.cos(dw_orientation[j])*np.sign(np.cos(y))
if dww=="SS":
    for j in range(size_z):
        mz[:,:,j]=np.cos(y)#np.abs(np.tanh((np.sin(y))/dww[j]))*np.sign(np.sin(y))
        m_inplane[:,:,j]=np.abs(np.sin(y))

        mx[:,:,j]=m_inplane[:,:,j]*np.sin(y/spiral_type[j])
        my[:,:,j]=m_inplane[:,:,j]*np.cos(y/spiral_type[j])
    
# mx2=np.zeros(mx.shape)
# for j in range(len(z)):
#     if j<=30:
#         mx2[:,:,j]=np.abs(mx[:,:,j])+x_bias
mx2=np.zeros((size_x,size_y*3,size_z))
my2=np.zeros((size_x,size_y*3,size_z))
mz2=np.zeros((size_x,size_y*3,size_z))

mx2[:,0:size_y,:]=mx
mx2[:,size_y:size_y*2,:]=mx
mx2[:,size_y*2:size_y*3,:]=mx

my2[:,0:size_y,:]=my
my2[:,size_y:size_y*2,:]=my
my2[:,size_y*2:size_y*3,:]=my

mz2[:,0:size_y,:]=mz
mz2[:,size_y:size_y*2,:]=mz
mz2[:,size_y*2:size_y*3,:]=mz

mx3=np.ndarray.flatten(mx2)
my3=np.ndarray.flatten(my2)
mz3=np.ndarray.flatten(mz2)

mx3=-mx3/np.sqrt(mx3**2+my3**2+mz3**2)
my3=my3/np.sqrt(mx3**2+my3**2+mz3**2)
mz3=mz3/np.sqrt(mx3**2+my3**2+mz3**2)

np.savetxt("../Magnetization Files/Mz_Analytic_crossover=5.csv", mz3, delimiter=",")

np.savetxt("../Magnetization Files/My_Analytic_crossover=5.csv", my3, delimiter=",")

np.savetxt("../Magnetization Files/Mx_Analytic_crossover=5.csv", mx3, delimiter=",")

#print(angle_dw)
#with open("LSMO_Simulation.py") as file:
#    exec(file.read())