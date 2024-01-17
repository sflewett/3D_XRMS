# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 15:24:44 2023

@author: sflewett
"""
import numpy as np
from pymatreader import read_mat

class LSMO():
    def __init__(self,full_column="full"):
        self.Incident=np.array([[complex(1,0)],[complex(0,1)]])#polarization basis of the incident light (sigma, pi)
        self.full_column=full_column
        #self.lamda=1.95e-9#Mn edge
        self.unit_cell=3.88e-10#unit cell size
        self.dx=2e-9
        self.dz=5e-9
        self.dz_mag=2.5e-9#thickness of the magnetic part of each layer
        files=["0my.mat","0mx.mat","0mz.mat"]#order is longitudinal, transverse, polar
        #files=["sky_my.mat","sky_mx.mat","sky_mz.mat"]#order is longitudinal, transverse, polar
        keys=["my01","mx01","mz01"]
        self.M=self.matlabread(files,keys)
        self.sigma_roughness=0e-10#rugosidad para el cálculo de la reflexión especular utilizando la aproximación de Born
        self.size_x=150
        self.size_y=200
        self.size_z=20#number of layers in the micromagnetic simulation
        self.size_z=40#inserting a non magnetic layer between each MM layer
        self.thickness=1e-7
        z=np.linspace(0,self.thickness,self.size_z+1)
        self.z=z#this is the z we use for calculating the attenuation at each layer in the 
        
        
        f_Mag=complex(-4.32,1.81)
        self.f_La=complex(24.6663,	8.24456	)
        self.f_Sr=complex(28.8485,	13.0226	)
        self.f_O=complex(6.02905,	3.75417)
        self.f_Mn=complex(-5.45,16.0)
        
        self.na_La=6162/0.138*6.0222e23
        self.na_Sr=2640/0.0876*6.0222e23
        self.na_O=1141/0.016*6.0222e23
        self.na_Mn=7210/0.054938*6.0222e23
        self.TMOKE_Threshold=0.995#values of the transverse magnetization above this value get processed using the algebra of equations 
        #39-44 of Stepanov Sinha
        self.M2=np.zeros((3,self.size_x,self.size_y,self.size_z+1),dtype=complex)
        self.na=np.zeros((self.size_x,self.size_y,self.size_z+1),dtype=complex)
        self.f_Charge=np.zeros((self.size_x,self.size_y,self.size_z+1),dtype=complex)
        self.f_Mag=np.zeros((self.size_x,self.size_y,self.size_z+1),dtype=complex)
        self.f_Mag2=np.zeros((self.size_x,self.size_y,self.size_z+1),dtype=complex)
        #M_muster=np.array([[[-np.sqrt(0),-np.sqrt(0),-np.sqrt(1)] for j in range(self.size_x)] for k in range(self.size_y)])
        #M_muster=np.transpose(M_muster, (2,0,1)) WE DONT NEED THESE BECAUSE WE HAVE MM SIMULATIONS
        for l in range(self.size_z+1):
            if l==0:
                self.M2[:,:,:,l]=np.zeros((3,self.size_x,self.size_y))
                self.na[:,:,l]=1.
                self.f_Charge[:,:,l]=1e-6
                self.f_Mag[:,:,l]=complex(0,0)
                self.f_Mag2[:,:,l]=complex(0,0)
                #we add the vacuum layer in the sample class
            if l%2==1:
                l2=(l-1)//2
                self.M2[:,:,:,l]=self.M[:,:,:,l2]
                self.na[:,:,l]=(self.na_O*2+self.na_Mn)/3
                self.f_Charge[:,:,l]=(self.f_O*2+self.f_Mn)/3
                self.f_Mag[:,:,l]=f_Mag
                self.f_Mag2[:,:,l]=complex(0,0)
            else:
                self.M2[:,:,:,l]=np.zeros((3,self.size_x,self.size_y))
                self.na[:,:,l]=(self.na_O+(self.na_La*0.666666+self.na_Sr*0.3333))/2
                self.f_Charge[:,:,l]=(self.f_O+(self.f_La*0.666666+self.f_Sr*0.3333))/2
                self.f_Mag[:,:,l]=0.
                self.f_Mag2[:,:,l]=0.
        self.M=self.M2
        
        density_adjustment=1.9#to set na near to tabulated value of 8.1e28
        
        self.na=self.na*density_adjustment
        
    def matlabread(self,files,keys):
        M=[]
        for j in range(3):
            data = read_mat(files[j])
            M.append(data[keys[j]])
        M=np.array(M)
        M2=np.zeros(M.shape,dtype=complex)
        mx=np.real(M[0,...])
        my=np.real(M[1,...])
        mz=np.real(M[2,...])
        mx2=mx
        my2=my
        mz2=mz
        # for j in range(mx2.shape[0]):
        #     for k in range(mx2.shape[1]):
        #         for l in range(mx2.shape[2]):
        #             if np.abs(my2[j,k,l])>0.99:
        #                 my2[j,k,l]=0.99*np.sign(my[j,k,l])
        #                 angle=np.arctan2(mz[j,k,l],mx[j,k,l])
        #                 m_trans=np.sqrt(1-0.99**2)
        #                 mz2[j,k,l]=m_trans*np.sin(angle)
        #                 mx2[j,k,l]=m_trans*np.cos(angle)
        M2[0,...]=mx
        M2[1,...]=my
        M2[2,...]=mz
        #this loop is to avoid a singlarity in the transverse MOKE case with the Stepanov
        #Sinha algorithm. In future, an exception needs to be incorporated into the main code
        #to avoid this unphysical workaround
        return(M2)        
    def interpolate_nearest_neighbour(self,R_array):
        #interpolating the array of reflection coefficients to get the final array
        #on which the diffraction is to be calculated
        thickness=self.thickness
        unit_cell=self.unit_cell
        periods=np.int32(thickness/unit_cell)
        R_even=R_array[:,:,::2,:,:]
        R_odd=R_array[:,:,1::2,:,:]
        R_interp_odd=np.zeros((R_array.shape[0],R_array.shape[1],np.int32(np.round(R_odd.shape[2]*self.dz/self.unit_cell)),R_array.shape[3],R_array.shape[4]),dtype=complex)
        R_interp_even=np.zeros((R_array.shape[0],R_array.shape[1],np.int32(np.round(R_even.shape[2]*self.dz/self.unit_cell)),R_array.shape[3],R_array.shape[4]),dtype=complex)
        for l in range(periods):
            index=np.int32(np.round(l/(self.dz/self.unit_cell)))
            if index<R_odd.shape[2]:
                R_interp_odd[:,:,l,:,:]=R_odd[:,:,index,:,:]
                R_interp_even[:,:,l,:,:]=R_even[:,:,index,:,:]
        if self.full_column=="full":
            R_interp=np.zeros((200,200,R_interp_odd.shape[2]+R_interp_even.shape[2],R_array.shape[3],R_array.shape[4]),dtype=complex)
            R_interp[25:175,:,::2,:,:]=R_interp_even#zero padded
            R_interp[25:175,:,1::2,:,:]=R_interp_odd
            R_interp[0:25,:,::2,:,:]=R_interp_even[125:150,:,:,:,:]#zero padded
            R_interp[0:25,:,1::2,:,:]=R_interp_odd[125:150,:,:,:,:]
            R_interp[175:200,:,::2,:,:]=R_interp_even[0:25,:,:,:,:]#zero padded
            R_interp[175:200,:,1::2,:,:]=R_interp_odd[0:25,:,:,:,:]
        else:
            R_interp=np.zeros((1,1,R_interp_odd.shape[2]+R_interp_even.shape[2],R_array.shape[3],R_array.shape[4]),dtype=complex)
            R_interp[:,:,::2,:,:]=R_interp_even#zero padded
            R_interp[:,:,1::2,:,:]=R_interp_odd
        self.z_interp=np.array(range(R_interp.shape[2]))*self.unit_cell/2
        self.R_interp=R_interp
        