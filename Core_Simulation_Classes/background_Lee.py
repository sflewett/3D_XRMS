# -*- coding: utf-8 -*-
#"""
#Created on Thu Sep 14 10:25:04 2023
#
#@author: sflewett
#"""
import numpy as np
from XRMS_Stepanov_Sinha import XRMS_Simulate
import XRMS_Simulation_Load

class Background_DWBA_Lee():
    def __init__(self, sample,simulation_input,theta_0):
        #here, the input is a range of exit angles, with the incidence angle assumed to be the median value of the input
        c=3e8
        energy=simulation_input['Simulation_Parameters']["energy"]
        f=energy*1.6e-19/6.626e-34
        self.lamda=c/f
        self.theta_0=theta_0
        
        det_pixel=simulation_input['Simulation_Parameters']['det_dx']
        detector_distance=simulation_input['Simulation_Parameters']["det_sample_distance"]
        n_det_x=simulation_input['Simulation_Parameters']['det_size'][0]
               
        if n_det_x%2==0:
            detector_xz=np.linspace(-n_det_x/2,n_det_x/2-1,n_det_x)*det_pixel
        else:
            detector_xz=np.linspace(-(n_det_x-1)/2,(n_det_x-1)/2,n_det_x)*det_pixel
        
        self.theta=np.arctan(detector_xz/detector_distance)+self.theta_0
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
        self.sigmas = simulation_input['Background_Parameters']["sigmas"]  
        self.eta_par = simulation_input['Background_Parameters']["eta_par"]
        self.h = simulation_input['Background_Parameters']["h"]       
        self.eta_perp=simulation_input['Background_Parameters']["eta_perp"]
        #These are parameters from the Lee 2003 paper
        self.specular_0=XRMS_Simulate(self.sample,self.theta_0,self.energy,full_column="column")
        #no point running the XRMS code in full 3D mode here, because we convolve with the magnetic scattering pattern afterwards.
        self.specular_0.Chi_define()
        self.specular_0.get_A_S_matrix()
        self.specular_0.XM_define()
        self.specular_0.Matrix_stack()
        self.u=np.squeeze(self.specular_0.U)
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
    
    def U_FFT(self,full_simple='full'):
        qz=self.qz
        qx=self.qx
        dim=qx.shape
        qx_central=qx[0,round(dim[1]/2)]
        delta_qx=qx[0,round(dim[1]/2)]-qx[0,round(dim[1]/2)+1]
        n=dim[1]
        delta_r=1/n/delta_qx
        if n%2==0:
            self.x = np.meshgrid(np.linspace(-n/2,n/2-1, n),np.linspace(-n/2,n/2-1, n))[1]
            self.y = np.meshgrid(np.linspace(-n/2,n/2-1, n),np.linspace(-n/2,n/2-1, n))[0]
            self.R = np.sqrt((self.y*delta_r*np.sin(self.theta_0))**2+(self.x*delta_r)**2)
        if n%2==1:
            n2=n-1
            self.x = np.meshgrid(np.linspace(-n2/2,n2/2, n),np.linspace(-n2/2,n2/2, n))[1]
            self.y = np.meshgrid(np.linspace(-n2/2,n2/2, n),np.linspace(-n2/2,n2/2, n))[0]
            self.R = np.sqrt((self.y*delta_r*np.sin(self.theta_0))**2+(self.x*delta_r)**2)
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
        #This and the following two methods are dedicated to filling the 3 lines of equation 5.12
        specular_0.chi_zero=np.squeeze(specular_0.chi_zero)
        specular_0.chi2=np.squeeze(specular_0.chi2)
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
        specular_0.chi_zero=np.squeeze(specular_0.chi_zero)
        specular_0.chi2=np.squeeze(specular_0.chi2)
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
        specular_0.chi_zero=np.squeeze(specular_0.chi_zero)
        specular_0.chi2=np.squeeze(specular_0.chi2)
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
        #evaluating equation 5.11 of the paper which is computed via equation 5.12, and calls the 3 previous methods
        theta_in=self.theta_0
        theta_out=self.theta
        self.U_FFT(full_simple='full')
        
        ntheta=len(self.theta)
        theta_i=min(self.theta)
        theta_f=max(self.theta)
        index_in=round((theta_in-theta_i)/(theta_f-theta_i)*(ntheta-1))
         
        qz=self.qz
        qx=self.qx
        Specular_Intensity=np.zeros((len(self.theta)))
        #also outputting the specular intensity
        diffuse_charge=np.zeros((len(theta_out),len(theta_out)),dtype=complex)
        diffuse_mag=np.zeros((len(theta_out),len(theta_out)),dtype=complex)
        diffuse_charge_mag=np.zeros((len(theta_out),len(theta_out)),dtype=complex)
        
        specular_0=self.specular_0
        specular_0.z=self.specular_0.z[1:]
        for index_out in range(1,ntheta-1):#why not from zero to ntheta??
            print(index_out)
            specular=XRMS_Simulate(self.sample,self.theta[index_out],self.energy,full_column="column")
            specular.Chi_define()
            specular.get_A_S_matrix()
            specular.XM_define()
            specular.Matrix_stack()
            Specular_Intensity[index_out]=specular.I_output[0]
            specular.z=specular.z[1:]
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

def background_main(simulation_input,theta):
        
    sample1=XRMS_Simulation_Load.Generic_sample(simulation_input,incident_beam_index=0)
    sample2=XRMS_Simulation_Load.Generic_sample(simulation_input,incident_beam_index=1)
    count=0
    
    count=count+1
    
    bkg1=Background_DWBA_Lee(sample1,simulation_input,theta)
    bkg2=Background_DWBA_Lee(sample2,simulation_input,theta)
    bkg1.get_q()
    bkg1.U_FFT()
    bkg1.BigSum()
    bkg2.get_q()
    bkg2.U_FFT()
    bkg2.BigSum()
    
    
    return bkg1,bkg2
    
