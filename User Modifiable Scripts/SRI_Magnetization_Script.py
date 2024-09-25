# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 09:09:18 2024

@author: sflewett
"""
import numpy as np

size_y=384
size_x=2
size_z=20
z=np.linspace(1,size_z,size_z)

dww=0.4*np.ones(len(z))#-0.00001*np.exp(-z)
dw_orientation=(np.tanh((z-7.0)/1)+1)*np.pi/2
#print(np.sin(dw_orientation))
#print(np.cos(dw_orientation))
skewness=1.2
#for angle_dw in np.linspace(np.pi/4,3*np.pi/8,6):
    #dw_orientation=angle_dw*np.ones(len(z))#(np.tanh((z+0)/1.6)+1)*np.pi/2
y=np.zeros(size_y)
for j in range(len(y)):
    jj=j%(size_y/3)
    if j ==0:
        y[j]=0
    if jj<size_y/6*skewness and j>0:
        dy=1/skewness
        y[j]=y[j-1]+dy
    if jj>=size_y/6*skewness and j>0:
        dy=size_y/6/(size_y/3-size_y/6*skewness)
        y[j]=y[j-1]+dy
#y=np.linspace(0,size_y-1,size_y)

y=y/size_y*6*np.pi
x_bias=0.0
mx=np.zeros((size_x,size_y,size_z))
my=np.zeros((size_x,size_y,size_z))
mz=np.zeros((size_x,size_y,size_z))

for j in range(size_z):
    if dww[j]!="SS":
        mz[:,:,j]=-np.abs(np.tanh((np.sin(y))/dww[j]))*np.sign(np.sin(y))
        mx[:,:,j]=np.abs((1-np.tanh((np.sin(y))/dww[j])**2)**0.5*np.sin(dw_orientation[j])*np.sign(np.cos(y)))+x_bias
        my[:,:,j]=-(1-np.tanh((np.sin(y))/dww[j])**2)**0.5*np.cos(dw_orientation[j])*np.sign(np.cos(y))
    if dww[j]=="SS":    
        mz[:,:,j]=np.sin(y)#np.abs(np.tanh((np.sin(y))/dww[j]))*np.sign(np.sin(y))
        mx[:,:,j]=np.abs(np.abs(1-np.sin(y)**2)**0.5*np.sin(dw_orientation[j])*(np.cos(y)))+x_bias
        my[:,:,j]=np.abs(1-np.sin(y)**2)**0.5*np.cos(dw_orientation[j])*np.sign(np.cos(y))
    
# mx2=np.zeros(mx.shape)
# for j in range(len(z)):
#     if j<=30:
#         mx2[:,:,j]=np.abs(mx[:,:,j])+x_bias
mx3=np.ndarray.flatten(mx-mx)
my3=np.ndarray.flatten(my)

mz3=np.ndarray.flatten(mz)
#print(mx3[0:10])
mx3=mx3/np.sqrt(mx3**2+my3**2+mz3**2)
my3=my3/np.sqrt(mx3**2+my3**2+mz3**2)
mz3=mz3/np.sqrt(mx3**2+my3**2+mz3**2)

np.savetxt("../Magnetization Files/SRI_z.csv", mz3, delimiter=",")

np.savetxt("../Magnetization Files/SRI_y.csv", my3, delimiter=",")

np.savetxt("../Magnetization Files/SRI_x.csv", mx3, delimiter=",")
#print(angle_dw)
#with open("LSMO_Simulation.py") as file:
#    exec(file.read())