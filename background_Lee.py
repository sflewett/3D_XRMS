# -*- coding: utf-8 -*-
#"""
#Created on Thu Sep 14 10:25:04 2023
#
#@author: sflewett
#"""
import numpy as np
from Stepanov_Specular import Stepanov_Specular
from LSMO_sample_class import LSMO
from Sample_Class_test import XRMS_Sample
import matplotlib.pyplot as plt

class Background_DWBA_Lee():
    def __init__(self, sample,theta,energy):
        #here, the input is a range of exit angles, with the incidence angle assumed to be the median value of the input
        c=3e8
        f=energy*1.6e-19/6.626e-34
        self.lamda=c/f
        self.theta=theta
        self.theta_0=np.median(theta)
        specular_list=[]
        self.energy=energy
        self.delta_theta=self.theta-self.theta_0
        self.f_charge=sample.f_Charge
        self.f_Mag=sample.f_Mag
        self.r0=2.82e-15
        self.na=sample.na
        self.M=sample.M
        self.size_x=sample.size_x
        self.size_y=sample.size_y
        self.sample=sample
        self.factor=self.na*self.r0*self.lamda**2/2/np.pi
        self.factor=np.sum(self.factor)/np.product(self.factor.shape)
        f_average=np.sum(self.f_charge)/np.product(self.f_charge.shape)
        self.n_average=1-self.factor*f_average
        self.sigmas = np.array([4.6e-10,4.6e-10])   
        self.eta_par = np.array([2.0e-8,2.0e-8]);
        self.h = np.array([0.3,0.3])        
        self.eta_perp=4.5e-8
        #These are parameters from the Lee 2003 paper
        self.specular_0=Stepanov_Specular(self.sample,self.theta_0,self.energy)
        self.specular_0.Chi_define()
        self.specular_0.get_A_S_matrix()
        self.specular_0.XM_define_Big()
        self.specular_0.Matrix_stack()
        self.u=self.specular_0.U
        #loading parameters from the incident wave
        
    def get_q(self):
        #this method evaluates the scattering vectors q according to equation 5.10 of the Lee paper
        u_pert=np.zeros((self.u.shape[0],self.u.shape[1],len(self.theta)),dtype=complex)
        u0=u_pert.copy()
        nx_pert=u_pert[0,:,:].copy()
        nx0=u0[0,:,:].copy()
        for j in range(len(self.theta)):
            temp=self.u/np.sin(self.theta_0)*np.sin(self.theta[j])
            #we just calculate u for the incident angle, and use approximate scaled values for the other angles. 
            #This will soon however be changed to include the explicit calculation of all u
            u_pert[:,:,j]=temp
            u0[:,:,j]=self.u
            nx_pert[:,j]=np.cos(self.theta[j])
            nx0[:,j]=np.cos(self.theta_0)
            
        for j in range (u_pert.shape[1]):
            if u_pert[2,j,0]==0:
                u_pert[2,j,:]=u_pert[1,j,:]
                u_pert[3,j,:]=u_pert[1,j,:]
                u_pert[1,j,:]=u_pert[0,j,:]
                u0[2,j,:]=u0[1,j,:]
                u0[3,j,:]=u0[1,j,:]
                u0[1,j,:]=u0[0,j,:]
                #in this part we are filling in the entries so that there are 4 entries for both the magnetic and non-magnetic elements
        #0 and 1 are for the incident beam, 2 and 3 for the reflected beam
        k=np.pi*2/self.lamda
        kz=k*u_pert
        kz0=k*u0
        kx=k*nx_pert
        kx0=k*nx0
        
        qz=np.zeros((4,self.u.shape[1],len(self.theta),2,2),dtype=complex)
        qx=kx-kx0   
        
        qz[0,:,:,0,0]=kz[0,:,:]-kz0[0,:,:]
        qz[0,:,:,0,1]=kz[0,:,:]-kz0[1,:,:]
        qz[0,:,:,1,0]=kz[1,:,:]-kz0[0,:,:]
        qz[0,:,:,1,1]=kz[1,:,:]-kz0[1,:,:]
        
        qz[1,:,:,0,0]=kz[0,:,:]-kz0[2,:,:]
        qz[1,:,:,0,1]=kz[0,:,:]-kz0[3,:,:]
        qz[1,:,:,1,0]=kz[1,:,:]-kz0[2,:,:]
        qz[1,:,:,1,1]=kz[1,:,:]-kz0[3,:,:]
        
        qz[2,:,:,0,0]=kz[2,:,:]-kz0[0,:,:]
        qz[2,:,:,0,1]=kz[2,:,:]-kz0[1,:,:]
        qz[2,:,:,1,0]=kz[3,:,:]-kz0[0,:,:]
        qz[2,:,:,1,1]=kz[3,:,:]-kz0[1,:,:]
        
        qz[3,:,:,0,0]=kz[2,:,:]-kz0[2,:,:]
        qz[3,:,:,0,1]=kz[2,:,:]-kz0[3,:,:]
        qz[3,:,:,1,0]=kz[3,:,:]-kz0[2,:,:]
        qz[3,:,:,1,1]=kz[3,:,:]-kz0[3,:,:]
        #indices: number, layer, angle, pol_in,pol_out
        #this is taken directly from equation 5.10
        self.qz=qz
        self.qx=qx
        
    def C_define(self,output=True):
        #ecuación 5,14 del paper
        sigma1=self.sigmas[0]
        sigma2=self.sigmas[1]
        R=self.R
        eta1_para=self.eta_par[0]
        eta2_para=self.eta_par[1]
        h1=self.h[0]
        h2=self.h[1]
        #the factor of z dependence is included in the variable "weights" in Bigsum
        C=sigma1*sigma2/2*(np.exp(-(np.abs(R)/eta1_para)**(2*h1))+
                           np.exp(-(np.abs(R)/eta2_para)**(2*h2)))
        self.C=C
        if output==True:
            return C
    
    def U_FFT(self,full_simple='simple'):
        qz=self.qz
        qx=self.qx
        dim=qx.shape
        qx_central=qx[0,round(dim[1]/2)]
        delta_qx=qx[0,round(dim[1]/2)]-qx[0,round(dim[1]/2)+1]
        n=dim[1]
        delta_r=1/n/delta_qx
        if n%2==0:
            self.x = np.meshgrid(np.linspace(-n/2,n/2-1, n),np.linspace(-n/2,n/2-1, n))[0]
            self.y = np.meshgrid(np.linspace(-n/2,n/2-1, n),np.linspace(-n/2,n/2-1, n))[1]
            self.R = np.sqrt(self.y**2+self.x**2)*delta_r
        if n%2==1:
            n2=n-1
            self.x = np.meshgrid(np.linspace(-n2/2,n2/2, n),np.linspace(-n2/2,n2/2, n))[0]
            self.y = np.meshgrid(np.linspace(-n2/2,n2/2, n),np.linspace(-n2/2,n2/2, n))[1]
            self.R = np.sqrt(self.y**2+self.x**2)*delta_r
        C=self.C_define()
        if n%2==0:
            qz_fixed=qz[1,0,round(n/2),0,0]
        if n%2==1:
            qz_fixed=qz[1,0,round((n-1)/2),0,0]
        integrand=np.exp(qz_fixed**2*C)-1
        if full_simple=='simple':
            integrand=C
        
            #here an important simplification is made, with the factors of q cancelling with multipliers in equation 5.11 
        U_temp=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(integrand)))
        self.U_FFT_arr=U_temp*np.exp((-1/2)*(qz_fixed**2*self.sigmas[0]**2+qz_fixed**2*self.sigmas[1]**2))
        if full_simple=='simple':
            self.U_FFT_arr=U_temp
            
    def fill_n_array_CC(self,Specular_Intensity,specular,specular_0,qz,index_out):
        T=np.array(specular.T_fields)
        R=np.array(specular.R_fields)
        T0=np.array(specular_0.T_fields)
        R0=np.array(specular_0.R_fields)
        efields=specular.efields
        efields0=specular_0.efields
        n_array=np.zeros((len(specular.z)-1),dtype=complex)
        n_prime_array=np.zeros((len(specular.z)-1),dtype=complex)
        for n in range(1,len(specular.z)):
             suma1=0
             for pol_in in range(2):#polarizations
                 for pol_out in range(2):
                     CC=[]
                     CC.append((T[n,pol_out])*T0[n,pol_in])
                     CC.append((T[n,pol_out])*R0[n,pol_in])
                     CC.append((R[n,pol_out])*T0[n,pol_in])
                     CC.append((R[n,pol_out])*R0[n,pol_in])
                     #equation 5.08
                     for p in range(4):                                  
                         if p==0:
                             e1_in=np.array(efields0[n,0,:])
                             e1_out=np.array(efields[n,0,:])
                             e2_in=np.array(efields0[n,1,:])
                             e2_out=np.array(efields[n,1,:])
                             e_in=[e1_in,e2_in]
                             e_out=[e1_out,e2_out]
                         if p==1:
                             e1_in=np.array(efields0[n,2,:])
                             e1_out=np.array(efields[n,0,:])
                             e2_in=np.array(efields0[n,3,:])
                             e2_out=np.array(efields[n,1,:])
                             e_in=[e1_in,e2_in]
                             e_out=[e1_out,e2_out]
                         if p==2:
                             e1_in=np.array(efields0[n,0,:])
                             e1_out=np.array(efields[n,2,:])
                             e2_in=np.array(efields0[n,1,:])
                             e2_out=np.array(efields[n,3,:])
                             e_in=[e1_in,e2_in]
                             e_out=[e1_out,e2_out]
                         if p==3:
                             e1_in=np.array(efields0[n,2,:])
                             e1_out=np.array(efields[n,2,:])
                             e2_in=np.array(efields0[n,3,:])
                             e2_out=np.array(efields[n,3,:])
                             e_in=[e1_in,e2_in]
                             e_out=[e1_out,e2_out]
                             #the electric fields in the sums  
                    
                         suma1=suma1+CC[p]*(np.abs(specular_0.chi_zero[n]-specular_0.chi_zero[n-1])**2).sum()*np.exp(-0.5*qz[p,n,index_out,pol_in,pol_out]**2*self.sigmas[0]**2)*\
                         np.dot(np.conj(e_out[pol_out]),(e_in[pol_in]))
             n_array[n-1]=suma1 
        for n_prime in range(1,len(specular.z)):
             suma1=0
             for pol_in in range(2):#polarizations
                 for pol_out in range(2):
                     CC=[]
                     CC.append((T[n_prime,pol_out])*T0[n_prime,pol_in])
                     CC.append((T[n_prime,pol_out])*R0[n_prime,pol_in])
                     CC.append((R[n_prime,pol_out])*T0[n_prime,pol_in])
                     CC.append((R[n_prime,pol_out])*R0[n_prime,pol_in])
                     for p in range(4):                                  
                         if p==0:
                             e1_in=np.array(efields[n_prime,0,:])
                             e1_out=np.array(efields0[n_prime,0,:])
                             e2_in=np.array(efields[n_prime,1,:])
                             e2_out=np.array(efields0[n_prime,1,:])
                             e_in=[e1_in,e2_in]
                             e_out=[e1_out,e2_out]
                         if p==1:
                             e1_in=np.array(efields[n_prime,2,:])
                             e1_out=np.array(efields0[n_prime,0,:])
                             e2_in=np.array(efields[n_prime,3,:])
                             e2_out=np.array(efields0[n_prime,1,:])
                             e_in=[e1_in,e2_in]
                             e_out=[e1_out,e2_out]
                         if p==2:
                             e1_in=np.array(efields[n_prime,0,:])
                             e1_out=np.array(efields0[n_prime,2,:])
                             e2_in=np.array(efields[n_prime,1,:])
                             e2_out=np.array(efields0[n_prime,3,:])
                             e_in=[e1_in,e2_in]
                             e_out=[e1_out,e2_out]
                         if p==3:
                             e1_in=np.array(efields[n_prime,2,:])
                             e1_out=np.array(efields0[n_prime,2,:])
                             e2_in=np.array(efields[n_prime,3,:])
                             e2_out=np.array(efields0[n_prime,3,:])
                             e_in=[e1_in,e2_in]
                             e_out=[e1_out,e2_out]                     
                    
                         suma1=suma1+np.conj(CC[p])*np.dot(np.conj(e_in[pol_in]),(e_out[pol_out]))*np.exp(-0.5*np.conj(qz[p,n_prime,index_out,pol_in,pol_out])**2*self.sigmas[0]**2)
                         #the exponentiated of qz come from before the integral in 5.13
             n_prime_array[n_prime-1]=suma1 
        return n_array, n_prime_array 
    
    def fill_n_array_MM(self,Specular_Intensity,specular,specular_0,qz,index_out):
        T=np.array(specular.T_fields)
        R=np.array(specular.R_fields)
        T0=np.array(specular_0.T_fields)
        R0=np.array(specular_0.R_fields)
        efields=specular.efields
        efields0=specular_0.efields
        n_array=np.zeros((len(specular.z)-1),dtype=complex)
        n_prime_array=np.zeros((len(specular.z)-1),dtype=complex)
        for n in range(1,len(specular.z)):
             suma1=0
             for pol_in in range(2):#polarizations
                 for pol_out in range(2):
                     CC=[]
                     CC.append((T[n,pol_out])*T0[n,pol_in])
                     CC.append((T[n,pol_out])*R0[n,pol_in])
                     CC.append((R[n,pol_out])*T0[n,pol_in])
                     CC.append((R[n,pol_out])*R0[n,pol_in])
                     #equation 5.08
                     for p in range(4):                                  
                         if p==0:
                             e1_in=np.array(efields0[n,0,:])
                             e1_out=np.array(efields[n,0,:])
                             e2_in=np.array(efields0[n,1,:])
                             e2_out=np.array(efields[n,1,:])
                             e_in=[e1_in,e2_in]
                             e_out=[e1_out,e2_out]
                         if p==1:
                             e1_in=np.array(efields0[n,2,:])
                             e1_out=np.array(efields[n,0,:])
                             e2_in=np.array(efields0[n,3,:])
                             e2_out=np.array(efields[n,1,:])
                             e_in=[e1_in,e2_in]
                             e_out=[e1_out,e2_out]
                         if p==2:
                             e1_in=np.array(efields0[n,0,:])
                             e1_out=np.array(efields[n,2,:])
                             e2_in=np.array(efields0[n,1,:])
                             e2_out=np.array(efields[n,3,:])
                             e_in=[e1_in,e2_in]
                             e_out=[e1_out,e2_out]
                         if p==3:
                             e1_in=np.array(efields0[n,2,:])
                             e1_out=np.array(efields[n,2,:])
                             e2_in=np.array(efields0[n,3,:])
                             e2_out=np.array(efields[n,3,:])
                             e_in=[e1_in,e2_in]
                             e_out=[e1_out,e2_out]
                             #the electric fields in the sums  
                         
                         ee=np.outer(np.conj(e_out[pol_out]),e_in[pol_in])
                         suma1=suma1+CC[p]*np.exp(-0.5*qz[p,n,index_out,pol_in,pol_out]**2*self.sigmas[1]**2)*\
                         np.sum(np.sum((specular_0.chi2[:,:,n]-specular_0.chi2[:,:,n-1])*ee))
             n_array[n-1]=suma1 
        for n_prime in range(1,len(specular.z)):
             suma1=0
             for pol_in in range(2):#polarizations
                 for pol_out in range(2):
                     CC=[]
                     CC.append((T[n_prime,pol_out])*T0[n_prime,pol_in])
                     CC.append((T[n_prime,pol_out])*R0[n_prime,pol_in])
                     CC.append((R[n_prime,pol_out])*T0[n_prime,pol_in])
                     CC.append((R[n_prime,pol_out])*R0[n_prime,pol_in])
                     for p in range(4):                                  
                         if p==0:
                             e1_in=np.array(efields[n_prime,0,:])
                             e1_out=np.array(efields0[n_prime,0,:])
                             e2_in=np.array(efields[n_prime,1,:])
                             e2_out=np.array(efields0[n_prime,1,:])
                             e_in=[e1_in,e2_in]
                             e_out=[e1_out,e2_out]
                         if p==1:
                             e1_in=np.array(efields[n_prime,2,:])
                             e1_out=np.array(efields0[n_prime,0,:])
                             e2_in=np.array(efields[n_prime,3,:])
                             e2_out=np.array(efields0[n_prime,1,:])
                             e_in=[e1_in,e2_in]
                             e_out=[e1_out,e2_out]
                         if p==2:
                             e1_in=np.array(efields[n_prime,0,:])
                             e1_out=np.array(efields0[n_prime,2,:])
                             e2_in=np.array(efields[n_prime,1,:])
                             e2_out=np.array(efields0[n_prime,3,:])
                             e_in=[e1_in,e2_in]
                             e_out=[e1_out,e2_out]
                         if p==3:
                             e1_in=np.array(efields[n_prime,2,:])
                             e1_out=np.array(efields0[n_prime,2,:])
                             e2_in=np.array(efields[n_prime,3,:])
                             e2_out=np.array(efields0[n_prime,3,:])
                             e_in=[e1_in,e2_in]
                             e_out=[e1_out,e2_out]                     
                    
                         ee=np.outer(np.conj(e_in[pol_in]),e_out[pol_out])
                         suma1=suma1+CC[p]*np.exp(-0.5*qz[p,n,index_out,pol_in,pol_out]**2*self.sigmas[1]**2)*\
                         np.sum(np.sum(np.conj(specular_0.chi2[:,:,n]-specular_0.chi2[:,:,n-1])*ee))
             n_prime_array[n_prime-1]=suma1 
        return n_array, n_prime_array 
    
    def fill_n_array_CM(self,Specular_Intensity,specular,specular_0,qz,index_out):
        T=np.array(specular.T_fields)
        R=np.array(specular.R_fields)
        T0=np.array(specular_0.T_fields)
        R0=np.array(specular_0.R_fields)
        efields=specular.efields
        efields0=specular_0.efields
        n_array=np.zeros((len(specular.z)-1),dtype=complex)
        n_prime_array=np.zeros((len(specular.z)-1),dtype=complex)
        for n in range(1,len(specular.z)):
             suma1=0
             for pol_in in range(2):#polarizations
                 for pol_out in range(2):
                     CC=[]
                     CC.append((T[n,pol_out])*T0[n,pol_in])
                     CC.append((T[n,pol_out])*R0[n,pol_in])
                     CC.append((R[n,pol_out])*T0[n,pol_in])
                     CC.append((R[n,pol_out])*R0[n,pol_in])
                     #equation 5.08
                     for p in range(4):                                  
                         if p==0:
                             e1_in=np.array(efields0[n,0,:])
                             e1_out=np.array(efields[n,0,:])
                             e2_in=np.array(efields0[n,1,:])
                             e2_out=np.array(efields[n,1,:])
                             e_in=[e1_in,e2_in]
                             e_out=[e1_out,e2_out]
                         if p==1:
                             e1_in=np.array(efields0[n,2,:])
                             e1_out=np.array(efields[n,0,:])
                             e2_in=np.array(efields0[n,3,:])
                             e2_out=np.array(efields[n,1,:])
                             e_in=[e1_in,e2_in]
                             e_out=[e1_out,e2_out]
                         if p==2:
                             e1_in=np.array(efields0[n,0,:])
                             e1_out=np.array(efields[n,2,:])
                             e2_in=np.array(efields0[n,1,:])
                             e2_out=np.array(efields[n,3,:])
                             e_in=[e1_in,e2_in]
                             e_out=[e1_out,e2_out]
                         if p==3:
                             e1_in=np.array(efields0[n,2,:])
                             e1_out=np.array(efields[n,2,:])
                             e2_in=np.array(efields0[n,3,:])
                             e2_out=np.array(efields[n,3,:])
                             e_in=[e1_in,e2_in]
                             e_out=[e1_out,e2_out]
                             #the electric fields in the sums  
                         
                         suma1=suma1+CC[p]*(specular_0.chi_zero[n]-specular_0.chi_zero[n-1]).sum()*np.exp(-0.5*qz[p,n,index_out,pol_in,pol_out]**2*self.sigmas[0]**2)*\
                         np.dot(np.conj(e_out[pol_out]),(e_in[pol_in]))
             n_array[n-1]=suma1 
        for n_prime in range(1,len(specular.z)):
             suma1=0
             for pol_in in range(2):#polarizations
                 for pol_out in range(2):
                     CC=[]
                     CC.append((T[n_prime,pol_out])*T0[n_prime,pol_in])
                     CC.append((T[n_prime,pol_out])*R0[n_prime,pol_in])
                     CC.append((R[n_prime,pol_out])*T0[n_prime,pol_in])
                     CC.append((R[n_prime,pol_out])*R0[n_prime,pol_in])
                     for p in range(4):                                  
                         if p==0:
                             e1_in=np.array(efields[n_prime,0,:])
                             e1_out=np.array(efields0[n_prime,0,:])
                             e2_in=np.array(efields[n_prime,1,:])
                             e2_out=np.array(efields0[n_prime,1,:])
                             e_in=[e1_in,e2_in]
                             e_out=[e1_out,e2_out]
                         if p==1:
                             e1_in=np.array(efields[n_prime,2,:])
                             e1_out=np.array(efields0[n_prime,0,:])
                             e2_in=np.array(efields[n_prime,3,:])
                             e2_out=np.array(efields0[n_prime,1,:])
                             e_in=[e1_in,e2_in]
                             e_out=[e1_out,e2_out]
                         if p==2:
                             e1_in=np.array(efields[n_prime,0,:])
                             e1_out=np.array(efields0[n_prime,2,:])
                             e2_in=np.array(efields[n_prime,1,:])
                             e2_out=np.array(efields0[n_prime,3,:])
                             e_in=[e1_in,e2_in]
                             e_out=[e1_out,e2_out]
                         if p==3:
                             e1_in=np.array(efields[n_prime,2,:])
                             e1_out=np.array(efields0[n_prime,2,:])
                             e2_in=np.array(efields[n_prime,3,:])
                             e2_out=np.array(efields0[n_prime,3,:])
                             e_in=[e1_in,e2_in]
                             e_out=[e1_out,e2_out]                     
                    
                         ee=np.outer(np.conj(e_in[pol_in]),e_out[pol_out])
                         suma1=suma1+CC[p]*np.exp(-0.5*qz[p,n,index_out,pol_in,pol_out]**2*self.sigmas[1]**2)*\
                         np.sum(np.sum(np.conj(specular_0.chi2[:,:,n]-specular_0.chi2[:,:,n-1])*ee))
             n_prime_array[n_prime-1]=suma1 
        return n_array, n_prime_array 
        
    def BigSum(self):
        #evaluating equation 5.11 of the paper
        theta_in=self.theta_0
        theta_out=self.theta
        self.U_FFT(full_simple='simple')
        
        ntheta=len(self.theta)
        theta_i=min(self.theta)
        theta_f=max(self.theta)
        index_in=round((theta_in-theta_i)/(theta_f-theta_i)*(ntheta-1))
         
        qz=self.qz
        qx=self.qx
        Specular_Intensity=np.zeros((len(theta)))
        #also outputting the specular intensity
        diffuse_charge=np.zeros((len(theta_out),len(theta_out)),dtype=complex)
        diffuse_mag=np.zeros((len(theta_out),len(theta_out)),dtype=complex)
        diffuse_charge_mag=np.zeros((len(theta_out),len(theta_out)),dtype=complex)
        
        specular_0=self.specular_0
        for index_out in range(1,ntheta-1):#why not from zero to ntheta??
            specular=Stepanov_Specular(self.sample,self.theta[index_out],self.energy)
            specular.Chi_define()
            specular.get_A_S_matrix()
            specular.XM_define_Big()
            specular.Matrix_stack()
            Specular_Intensity[index_out]=specular.I_output[0]
            if index_out==1:
                weights=np.array([[np.exp(-np.abs(specular.z[i]-specular.z[j])/self.eta_perp) for i in range(1,len(specular.z))] for j in range(1,len(specular.z))]);
            #these weights come from the definition of C (equation 5.14)
            n_array,n_prime_array=self.fill_n_array_CC(Specular_Intensity,specular,specular_0,qz,index_out)
            n_matrix_charge=np.outer(n_array,n_prime_array)*weights
            diffuse_charge[index_out,:]=np.sum(n_matrix_charge)*self.U_FFT_arr[index_out,:]
            
            n_array,n_prime_array=self.fill_n_array_MM(Specular_Intensity,specular,specular_0,qz,index_out)
            n_matrix_mag=np.outer(n_array,n_prime_array)*weights
            diffuse_mag[index_out,:]=np.sum(n_matrix_mag)*self.U_FFT_arr[index_out,:]
            
            n_array,n_prime_array=self.fill_n_array_CM(Specular_Intensity,specular,specular_0,qz,index_out)
            n_matrix_charge_mag=np.outer(n_array,n_prime_array)*weights
            diffuse_charge_mag[index_out,:]=np.sum(n_matrix_charge_mag)*self.U_FFT_arr[index_out,:]
            #breakpoint()
        self.diffuse_background=diffuse_charge+diffuse_mag+diffuse_charge_mag
        self.SI=Specular_Intensity
        #Including the final two magnetic terms in the background calculation is still pending
        #however these factors are of a substantially lesser magnitude compared with the charge scattering
        #calulated here
        
sample1=XRMS_Sample(circular="left")
sample2=XRMS_Sample(circular="right")
count=0
for angletest in range(1):
    count=count+1
    print(count)
    theta_i=13.0+angletest; #Angulo inicial para la lista de angulos
    theta_f=18.0+angletest; #Angulo final  
    theta_i=theta_i/180*np.pi
    theta_f=theta_f/180*np.pi
    ntheta=500
    theta=np.linspace(theta_i,theta_f,ntheta)        
    bkg1=Background_DWBA_Lee(sample1,theta,energy=780)
    bkg2=Background_DWBA_Lee(sample2,theta,energy=780)
    bkg1.get_q()
    bkg1.U_FFT()
    bkg1.BigSum()
    bkg2.get_q()
    bkg2.U_FFT()
    bkg2.BigSum()
    plt.figure()
    #plt.plot(theta[1:],np.log(np.abs(bkg.diffuse_background[1:])))
    plt.imshow(np.log(np.real(bkg1.diffuse_background[1:,1:])))
    test=np.zeros((500,500))

    test[250,250]=1

    test[250,200]=0.05

    test[250,300]=0.02
    background=bkg1.diffuse_background
    test2=np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(np.fft.fft2(test)*np.fft.fft2(background))))
    test2=np.abs(test2)
    test2[235:265,235:265]=test2[10,10]
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.xlabel("transverse deviation (deg)")
    plt.ylabel("longitudinal/polar deviation (deg)")
    image = np.transpose(np.log(test2[1:-1,1:-1]))
    i = ax.imshow(image, interpolation='nearest',
              extent=[0., 5., 0., 5])
    
