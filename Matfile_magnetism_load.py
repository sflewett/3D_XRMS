# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:01:56 2023

@author: sflewett
"""

from pymatreader import read_mat
import numpy as np

class element_properties:
    def __init__(self,element,energy):
        #these parameters are in general energy dependent
        #Please insert elemental or mixture parameters
        if element=="Co":
            self.f_Henke=complex(-18.6574,68.7554)
            self.f_Mag=complex(-2.32,-17.81)
            self.na=8900/0.0589*6.0222e23
        self.factor=self.na_Co*self.r0*self.lamda**2/2/np.pi
        self.n=1-self.factor_Co*self.f_Henke
        
        'Add the other elemental data in here'
        
class sample:
    def __init__(self):
        self.p_LaSrO=element("LaSrO")
        self.Mn==element("MnO")
        
        self.lamda=1.95e-9#Mn edge
        self.d=3.88e-10
        self.Mdx=2e-9
        self.Mdz=5e-9
        files=["0mx.mat","0my.mat","0mz.mat"]
        keys=["mx01","my01","mz01"]
        self.M=matlabread(files,keys)
        self.z=np.linspace(0,100e-9,100e-9/(self.d/2))
        self.layers=[]
        for j in range(len(self.z)):
            if j%2==0:
                self.layers.append("LaSrO")
            if j%2==1:
                self.layers.append("MnO")
        self.thicknesses=np.diff(self.z)
        
class XRMS_Simulate:
    def __init__(self,sample):        
        super().__init__(sample)
    
    def Chi_define(self,sample):
        #Chi as defined in the Stepanov Sinha Paper
        #X:longitudinal
        #Y: Transverse
        #Z: polar
        #na es la densidad de átomos
        M=sample.M
        if type(M)==int or type(M)==float:
            if M==0:
                M=np.array([0.,0.,0.])
       
        multiplier=self.lamda**2*self.r0/np.pi
        if np.array(M).sum()==0:
            chi_zero=na*multiplier*f_Henke#check the sign of this part
        else:
            f_Charge2=complex(-18.45,18.91)#for adjusting the refractive index of the Co layer
            chi_zero=na*multiplier*f_Charge
        B=na*multiplier*f_Mag
        C=0#This needs to be set in the case that a quadratic term is present
       #And should be set in the initialization routine as a global parameter
        chi=np.zeros((3, 3),dtype=complex)
        epsilon = np.array([[[int((i - j) * (j - k) * (k - i) / 2) for k in range(3)] for j in range(3)] for i in range(3)])
        delta=np.identity(3)
        MM = np.array([[M[j]*M[i] for i in range(3)] for j in range(3)]);
        temp=np.array([epsilon[:,:,k]*M[k] for k in range(3)]).sum(axis=0) 
        if split==False:
            chi = (chi_zero)*delta\
                -complex(0,1)*B*temp+C*MM
            return chi, chi_zero
        elif split==True:
            chi1 = (chi_zero)*delta
            chi2 = -complex(0,1)*B*temp+C*MM
            return chi1,chi2,chi_zero
        
        






    