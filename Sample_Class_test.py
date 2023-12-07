# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:45:28 2023

@author: sflewett
"""
import numpy as np
class XRMS_Sample():
    def __init__(self,circular):
        if circular=="left":
            self.Incident=np.array([[complex(1,0)],[-complex(0,1)]])#polarization basis of the incident light (sigma, pi)
        if circular=="right":
            self.Incident=np.array([[complex(1,0)],[complex(0,1)]])#polarization basis of the incident light (sigma, pi)
        self.size_x=256
        self.size_y=256
        self.size_z=32
        self.sigma_roughness=0e-10#rugosidad para el cálculo de la reflexión especular utilizando la aproximación de Born
        self.dx=5e-9
        self.dy=5e-9
        self.dz=1e-9
        self.dz_mag=5e-10#thickness of the magnetic part of each layer
        z=np.linspace(0,20e-8,self.size_z)
        z_random=np.random.normal(0, 6e-10, len(z))
        self.z=z+z_random
        #-Datos para prueba-#
        self.f_Henke_Co=complex(-16.6574,16.7554)#From the Henke website
        self.f_Charge=complex(-18.45,18.91)#The original value is 68.91!!!   From the beamline calibration, including both the non-resonant and resonant effects
        self.f_Mag=complex(-2.32,-17.81)
        self.f_Henke_Al=complex(11.9834,	1.15872)/1
        self.f_Henke_Pt=complex(44.2542,	27.0584)/1
        self.f_Henke_Ta=complex(43.1425,   20.1243)/1
        self.na_Fe=7874/0.0558*6.0222e23
        self.na_Gd=7900/0.157*6.0222e23
        self.na_Co=8900/0.0589*6.0222e23
        self.na_Al=2700/0.02698*6.0222e23
        
        self.M=np.zeros((3,self.size_x,self.size_y,self.size_z),dtype=complex)
        self.na=np.zeros((self.size_x,self.size_y,self.size_z),dtype=complex)
        self.f_Henke=np.zeros((self.size_x,self.size_y,self.size_z),dtype=complex)
        self.f_Charge=np.zeros((self.size_x,self.size_y,self.size_z),dtype=complex)
        self.f_Mag=np.zeros((self.size_x,self.size_y,self.size_z),dtype=complex)
        self.f_Mag2=np.zeros((self.size_x,self.size_y,self.size_z),dtype=complex)
        M_muster=np.array([[[-np.sqrt(0),-np.sqrt(0),-np.sqrt(1)] for j in range(self.size_x)] for k in range(self.size_y)])
        M_muster=np.transpose(M_muster, (2,0,1))
        for l in range(self.size_z):
            if l%2==0:
                self.M[:,:,:,l]=M_muster
                self.na[:,:,l]=8900/0.0589*6.0222e23
                self.f_Charge[:,:,l]=complex(-18.45,8.91)
                self.f_Mag[:,:,l]=complex(-2.32,-17.81)
                self.f_Mag2[:,:,l]=complex(0,0)
            else:
                self.M[:,:,:,l]=M_muster-M_muster
                self.na[:,:,l]=21450/0.195*6.0222e23
                self.f_Charge[:,:,l]=complex(44.2542,	27.0584)
                self.f_Mag[:,:,l]=0.
                self.f_Mag2[:,:,l]=complex(0,0)
            
        
        