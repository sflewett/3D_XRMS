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

dww=0.3
dw_orientation=(np.tanh((z-8)/5)+1)*np.pi/2

y=np.linspace(0,size_y-1,size_y)

y=y/size_y*6*np.pi
x_bias=0.5
mx=np.zeros((size_x,size_y,size_z))
my=np.zeros((size_x,size_y,size_z))
mz=np.zeros((size_x,size_y,size_z))

for j in range(size_z):
    mx[:,:,j]=(1-np.tanh((np.sin(y))/dww)**2)**0.5*np.sin(dw_orientation[j])*np.sign(np.cos(y))
    my[:,:,j]=(1-np.tanh((np.sin(y))/dww)**2)**0.5*np.cos(dw_orientation[j])*np.sign(np.cos(y))
    mz[:,:,j]=np.abs(np.tanh((np.sin(y))/dww))*np.sign(np.sin(y))

mx3=np.abs(np.ndarray.flatten(mx))+x_bias

my3=np.ndarray.flatten(my)

mz3=np.ndarray.flatten(mz)

mx3=mx3/np.sqrt(mx3**2+my3**2+mz3**2)
my3=my3/np.sqrt(mx3**2+my3**2+mz3**2)
mz3=mz3/np.sqrt(mx3**2+my3**2+mz3**2)

np.savetxt("Mz_Analytic_crossover=8.csv", mz3, delimiter=",")

np.savetxt("My_Analytic_crossover=8.csv", my3, delimiter=",")

np.savetxt("Mx_Analytic_crossover=8.csv", mx3, delimiter=",")