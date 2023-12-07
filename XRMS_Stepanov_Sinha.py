# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 09:40:32 2023

@author: sflewett
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from numba import njit
#from Sample_Class_test import XRMS_Sample
from Stepanov_Specular import Stepanov_Specular
from LSMO_sample_class import LSMO
#import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ffmpeg import FFmpeg, Progress
#Set this all up so that intrinsically 1d input arrays are only input as 1d arrays
#look up __all__ statement
#read up on class inheritance
#THIS CODE IS CURRENTLY SET UP SO THAT THE MAGNETIC INPUT IS 3D

class XRMS_Simulate():
    def __init__(self, sample,specular,theta,energy):
        
        c=3e8
        f=energy*1.6e-19/6.626e-34
        self.lamda=c/f
        self.theta=theta
        self.f_charge=sample.f_Charge
        self.f_Mag=sample.f_Mag
        self.f_Mag2=sample.f_Mag2
        self.r0=2.82e-15
        self.na=sample.na
        self.M=sample.M
        self.size_x=sample.size_x
        self.size_y=sample.size_y
        self.T_fields_linear=specular.T_fields_linear
        self.sample=sample
        self.factor=self.na*self.r0*self.lamda**2/2/np.pi
        self.factor=np.sum(self.factor)/np.product(self.factor.shape)
        f_average=np.sum(self.f_charge)/np.product(self.f_charge.shape)
        self.n_average=1-self.factor*f_average
        
    def Chi_define(self):
        #M is set up as [m_vector,xpos,ypos,zpos]
        #Chi as defined in the Stepanov Sinha Paper
        #X:longitudinal
        #Y: Transverse
        #Z: polar
        #na es la densidad de átomos
        lamda=self.lamda
        f_charge=self.f_charge
        f_Mag=self.f_Mag
        f_Mag2=self.f_Mag2
        r0=self.r0
        na=self.na
        M=self.M
        
        multiplier=lamda**2*r0/np.pi
        chi_zero=na*multiplier*f_charge
        B=na*multiplier*f_Mag
        C=na*multiplier*f_Mag2#This needs to be set in the case that a quadratic term is present
       #And should be set in the initialization routine as a global parameter
        chi=np.zeros((3, 3),dtype=complex)
       
        
        m1=np.array(M[0,...])
        m2=np.array(M[1,...])
        m3=np.array(M[2,...])
        empty=np.zeros(m1.shape)
        ones=empty+1
        delta=np.array([[ones,empty,empty],
                        [empty,ones,empty],
                        [empty,empty,ones]])
        m_epsilon1=np.array([[empty,empty,empty],
                   [empty,empty,m1],
                   [empty,-m1,empty]])
        m_epsilon2=np.array([[empty,empty,-m2],
                   [empty,empty,empty],
                   [m2,empty,empty]])
        m_epsilon3=np.array([[empty,m3,empty],
                   [-m3,empty,empty],
                   [empty,empty,empty]])
        temp=m_epsilon1+m_epsilon2+m_epsilon3
        MM = np.array([[M[j,...]*M[i,...] for i in range(3)] for j in range(3)])
        chi = (chi_zero)*delta\
            -complex(0,1)*B*temp+C*MM
        self.chi=chi
        self.chi_zero=chi_zero*ones
        
        
        
    def get_U(self):
        
        M=self.M
        chi=self.chi
        chi_zero=self.chi_zero
        theta=self.theta
        
        nx=np.array(np.cos(theta),dtype=complex);
        gamma=np.array(np.sin(theta),dtype=complex);
        M_tot=(M**2).sum(axis=0)
        M_tot2=M_tot
        dim1=M_tot2.shape[0]
        dim2=M_tot2.shape[1]
        dim3=M_tot2.shape[2]
        dim=np.array([dim1,dim2,dim3],dtype=np.int64)
        
        mask1=M_tot!=0
        mask2=M_tot==0
        #import quartic_solver_analytic
        
        @njit
        def Uloop(M_tot,Q1,Q2,Q3,Q4,Q5,output_arrays):
            u1=output_arrays[0]
            u2=output_arrays[1]
            u3=output_arrays[2]
            u4=output_arrays[3]
            for j in range(dim[0]):
                for k in range(dim[1]):
                     for l in range(dim[2]):
                         if M_tot[j,k,l]!=0:
                             q=np.array([Q1[j,k,l],Q2[j,k,l],Q3[j,k,l],Q4[j,k,l],Q5[j,k,l]])
                             temp1,temp2,temp3,temp4=np.roots(q)                                        
                             u1[j,k,l],u2[j,k,l],u3[j,k,l],u4[j,k,l]=temp1,temp2,temp3,temp4
            return u1,u2,u3,u4
        
        
        def Umag(chi,nx,gamma):
            u1=np.zeros((dim),dtype=np.float64)
            u2=u1
            u3=u1
            u4=u1
            Q1=1+chi[2,2,...]#[2,2,...]
            Q2=nx*(chi[0,2,...]+chi[2,0,...])
            Q3=chi[0,2,...]*chi[2,0,...]+chi[1,2,...]*chi[2,1,...]-(1+chi[2,2,...])*(gamma**2+chi[1,1,...])-(1+chi[0,0,...])*(gamma**2+chi[2,2,...])
            Q4=nx*(chi[0,1,...]*chi[1,2,...]+chi[1,0,...]*chi[2,1,...]-(chi[0,2,...]+chi[2,0,...])*(gamma**2+chi[1,1,...]))  
            Q5=(1+chi[0,0,...])*((gamma**2+chi[1,1,...])*(gamma**2+chi[2,2,...])-chi[1,2,...]*chi[2,1,...])-\
            chi[0,1,...]*chi[1,0,...]*(gamma**2+chi[2,2,...])- chi[0,2,...]*chi[2,0,...]*(gamma**2+chi[1,1,...])+\
            chi[0,1,...]*chi[1,2,...]*chi[2,0,...]+chi[1,0,...]*chi[2,1,...]*chi[0,2,...]
            
            output_arrays=np.array([u1,u2,u3,u4],dtype=np.complex128)                    
            u1,u2,u3,u4=Uloop(M_tot,Q1,Q2,Q3,Q4,Q5,output_arrays)
            u=np.array([u1,u2,u3,u4],dtype=complex)
            
            imag=u*complex(0,1)
            u_sorted=np.sort(imag,axis=0)
            u_sorted=u_sorted/complex(0,1)
            temp=np.zeros((u_sorted.shape),dtype=complex)
            temp[0:2,:,:,:]=u_sorted[0:2,:,:,:]
            temp[2,:,:,:]=u_sorted[2,:,:,:]
            temp[3,:,:,:]=u_sorted[3,:,:,:]
            u_sorted=temp
            u=u_sorted
            D=(chi[0,2,...]+u*nx)*(chi[2,0,...]+u*nx)-(1-u**2+chi[0,0,...])*(gamma**2+chi[2,2,...])
            Px=(chi[0,1,...]*(gamma**2+chi[2,2,...])-chi[2,1,...]*(chi[0,2,...]+u*nx))/D
            # we calculate the eigenvector here as a means for sorting the order of the eigenvalues, and therefore the filling of the matrices
            u_i=u[0:2,:,:,:]
            u_r=u[2:4,:,:,:]
            
            Px_i=np.imag(Px[0:2,:,:,:])
            Px_r=np.imag(Px[2:4,:,:,:])
            
            sort_ind_i=np.argsort(Px_i,axis=0)
            u_i_sorted=np.take_along_axis(u_i,sort_ind_i,axis=0)
            
            sort_ind_r=np.argsort(Px_r,axis=0)
            u_r_sorted=np.take_along_axis(u_r,sort_ind_r,axis=0)
            
            u[0:2,:,:,:]=u_i_sorted
            u[2,:,:,:]=u_r_sorted[1,:,:,:]
            u[3,:,:,:]=u_r_sorted[0,:,:,:]
            # the sorting is done such that there are no sudden phase jumps in Px and Pz
            return u
        def Unonmag(chi_zero,nx,gamma):# non magnetic case
            u1=(chi_zero+gamma**2)**0.5
            u2=-u1
            u3=u2-u2#setting these to zero
            u4=u3
            u=np.array([u1,u2,u3,u4],dtype=complex)
            
            return u
        U_dims=[4]+list(M_tot.shape) 
        U=np.zeros((U_dims),dtype=complex)
        U[:,mask1]=np.array(Umag(chi,nx,gamma))[:,mask1]
        U[:,mask2]=np.array(Unonmag(chi_zero,nx,gamma))[:,mask2]
        self.U=U
        #Ecuaciones 25-30 de Stepanov Sinha
    
    def get_A_S_matrix(self):#medium boundary matrix
         M=self.M
         chi=self.chi
         chi_zero=self.chi_zero
         theta=self.theta
         nx=np.array(np.cos(theta),dtype=complex);
         gamma=np.array(np.sin(theta),dtype=complex);
         M_tot=(M**2).sum(axis=0)
         mask1=M_tot!=0
         mask2=M_tot==0
         def Small_matrix_nomag(chi,chi_zero,theta):
             u=self.U
             epsilon=1+chi_zero
             ones=np.ones(M_tot.shape)
             zeros=np.zeros(M_tot.shape)
             Matrix=[[ones,zeros,ones,zeros],\
                    [zeros,epsilon**0.5,zeros,epsilon**0.5],\
                    [u[0,...],zeros,u[1,...],zeros],\
                    [zeros,u[0,...]/epsilon**0.5,zeros,u[1,...]/epsilon**0.5]]
                 
             P_dims=[4]+list(M_tot.shape) 
             Px=np.zeros((P_dims))
             Pz=Px
             return np.array(Matrix,dtype=complex),Px,Pz
         def Small_matrix_mag(chi,chi_zero,theta,nx,gamma):
             u=self.U
             D=(chi[0,2,...]+u*nx)*(chi[2,0,...]+u*nx)-(1-u**2+chi[0,0,...])*(gamma**2+chi[2,2,...])
             Px=(chi[0,1,...]*(gamma**2+chi[2,2,...])-chi[2,1,...]*(chi[0,2,...]+u*nx))/D
             Pz=(chi[2,1,...]*(1-u**2+chi[0,0,...])-chi[0,1,...]*(chi[2,0,...]+u*nx))/D
             
             
             #D_test=chi[1,0,...]*(gamma**2+chi[2,2,...])-chi[1,2,...]*(chi[2,0,...]+u*nx)
             #Px_test=((gamma**2+chi[1,1,...]-u**2)*(gamma**2+chi[2,2,...])-chi[1,2,...]*chi[2,1,...])/D_test
             
             
             
             v=u*Px-nx*Pz
             w=Px
             ones=np.ones(M_tot.shape,dtype=complex)
             Matrix=[[ones,ones,ones,ones],\
                     [v[0,...],v[1,...],v[2,...],v[3,...]],\
                     [u[0,...],u[1,...],u[2,...],u[3,...]],\
                     [w[0,...],w[1,...],w[2,...],w[3,...]]]
             
             return np.array(Matrix),Px,Pz
         
         A_S_matrix_dims=[4,4]+list(M_tot.shape) 
         A_S_matrix=np.zeros((A_S_matrix_dims),dtype=complex)
         P_dims=[4]+list(M_tot.shape) 
         self.Px=np.zeros((P_dims),dtype=complex)
         self.Pz=np.zeros((P_dims),dtype=complex)
         self.get_U()
         temp1,temp2,temp3=Small_matrix_mag(chi,chi_zero,theta,nx,gamma)
         A_S_matrix[:,:,mask1],self.Px[:,mask1],self.Pz[:,mask1]=temp1[:,:,mask1],temp2[:,mask1],temp3[:,mask1]
         temp1,temp2,temp3=Small_matrix_nomag(chi,chi_zero,theta)
         A_S_matrix[:,:,mask2],self.Px[:,mask2],self.Pz[:,mask2]=temp1[:,:,mask2],temp2[:,mask2],temp3[:,mask2]
         A_S_matrix2=np.zeros((A_S_matrix.shape[0],A_S_matrix.shape[1],A_S_matrix.shape[2],A_S_matrix.shape[3],A_S_matrix.shape[4]+1),dtype=complex)
         ones=np.ones(M_tot[:,:,0].shape)
         zeros=np.zeros(M_tot[:,:,0].shape)
         Matrix=np.array([[ones,zeros,ones,zeros],\
                [zeros,ones,zeros,ones],\
                [ones*gamma,zeros,-ones*gamma,zeros],\
                [zeros,ones*gamma,zeros,-ones*gamma]])
         A_S_matrix2[...,1:A_S_matrix.shape[-1]+1]=A_S_matrix
         A_S_matrix2[...,0]=Matrix
         self.AS_Matrix=A_S_matrix2
         Px2=np.zeros((self.Px.shape[0],self.Px.shape[1],self.Px.shape[2],self.Px.shape[3]+1),dtype=complex)
         Pz2=np.zeros((self.Px.shape[0],self.Px.shape[1],self.Px.shape[2],self.Px.shape[3]+1),dtype=complex)
         Px2[...,1:self.Px.shape[-1]+1]=self.Px
         Pz2[...,1:self.Pz.shape[-1]+1]=self.Pz
         self.Px=Px2
         self.Pz=Pz2
     
    def XM_define_Small(self): 
         theta=self.theta
         A_S_Matrix=self.AS_Matrix
         M=self.M
         Px=self.Px
         Pz=self.Pz
         AS1=A_S_Matrix[...,0:A_S_Matrix.shape[-1]-1]#upper
         AS2=A_S_Matrix[...,1:A_S_Matrix.shape[-1]]#lower
         M2=np.zeros((M.shape[0],M.shape[1],M.shape[2],M.shape[3]+1),dtype=complex)
         M2[:,:,:,1:M2.shape[-1]]=M
         M=M2
         M_tot=(M**2).sum(axis=0)
         mask1=M_tot!=0
         mask2=M_tot==0
         mask1=mask1[...,0:M_tot.shape[-1]-1]
         mask2=mask2[...,0:M_tot.shape[-1]-1]
         if len(A_S_Matrix.shape)==3:
             permutation=(2,0,1)
         if len(A_S_Matrix.shape)==4:
             permutation=(2,3,0,1)
         if len(A_S_Matrix.shape)==5:
             permutation=(2,3,4,0,1)
         AS1=np.transpose(AS1, permutation)
         AS2=np.transpose(AS2, permutation)
        
         X=np.matmul(np.linalg.inv(AS1),AS2)
         Xtt=X[...,0:2,0:2]
         Xrt=X[...,2:4,0:2]
         #@jit
         def Mrt_nomag_function(Xtt,Xrt,mask): 
             Mrt=np.matmul(Xrt,np.linalg.inv(Xtt))
             return Mrt
         #@njit
         def Mrt_mag(Xtt,Xrt,M_tot):
             #M2=M_tot[...,0:M_tot.shape[-1]-1]
             Px2=Px[...,0:Px.shape[-1]-1]
             #Px3=Px2[:,mask]
             Pz2=Pz[...,0:Pz.shape[-1]-1]
             #Pz3=Pz2[:,mask]
             Basischange=Xrt-Xrt
             ones=np.ones(M_tot[:,:,0:-1].shape)
             zeros=ones-ones
             Basischange[...,0,0]=ones
             Basischange[...,0,1]=ones
             Basischange[...,1,0]=Px2[0,...]*np.sin(theta)+Pz2[0,...]*np.cos(theta)
             Basischange[...,1,1]=Px2[1,...]*np.sin(theta)+Pz2[1,...]*np.cos(theta)
             Basischange2=Basischange
             Basischange2[...,1,0]=Px2[2,...]*np.sin(theta)+Pz2[2,...]*np.cos(theta)
             Basischange2[...,1,1]=Px2[3,...]*np.sin(theta)+Pz2[3,...]*np.cos(theta)
            
             Basischange=np.array(Basischange)
             Basischange2=np.array(Basischange2)
             ones=np.ones(M_tot[:,:,0].shape)
             zeros=ones-ones
             for j in range(Basischange.shape[2]):
                 if np.sum(M_tot[:,:,j])==0:
                     Basischange[:,:,j,0,0]=ones
                     Basischange[:,:,j,0,1]=zeros
                     Basischange[:,:,j,1,0]=zeros
                     Basischange[:,:,j,1,1]=ones
                     Basischange2[:,:,j,0,0]=ones
                     Basischange2[:,:,j,0,1]=zeros
                     Basischange2[:,:,j,1,0]=zeros
                     Basischange2[:,:,j,1,1]=ones
             
             #@jit
             def Matrixalgebra(Basischange,Basischange2,Xrt,Xtt):
                 Eigen_reflection=np.matmul(Xrt,np.linalg.inv(Xtt))         
                 processed_matrices=np.matmul(np.matmul(Basischange2,Eigen_reflection),np.linalg.inv(Basischange))
                 return processed_matrices
             Mrt=Matrixalgebra(Basischange,Basischange2,Xrt,Xtt)
             return Mrt
         
         
         self.Mrt_matrix=Mrt_mag(Xtt,Xrt,M_tot)
         # Mrt_mag=np.zeros((self.size_x,self.size_y,int(temp.shape[0]/(self.size_x*self.size_y)),2,2),dtype=complex)
         # Mrt_nomag=np.zeros((self.size_x,self.size_y,int(temp.shape[0]/(self.size_x*self.size_y)),2,2),dtype=complex)
         # Mrt_mag[:,:,:,0,0]=temp[:,0,0].reshape(self.size_x,self.size_y,int(temp.shape[0]/(self.size_x*self.size_y)),order="C")
         # Mrt_mag[:,:,:,0,1]=temp[:,0,1].reshape(self.size_x,self.size_y,int(temp.shape[0]/(self.size_x*self.size_y)),order="C")
         # Mrt_mag[:,:,:,1,0]=temp[:,1,0].reshape(self.size_x,self.size_y,int(temp.shape[0]/(self.size_x*self.size_y)),order="C")
         # Mrt_mag[:,:,:,1,1]=temp[:,1,1].reshape(self.size_x,self.size_y,int(temp.shape[0]/(self.size_x*self.size_y)),order="C")
         
         # temp=Mrt_nomag_function(Xtt[mask2,:,:],Xrt[mask2,:,:],mask2)
         # Mrt_nomag[:,:,:,0,0]=temp[:,0,0].reshape(self.size_x,self.size_y,int(temp.shape[0]/(self.size_x*self.size_y)),order="C")
         # Mrt_nomag[:,:,:,0,1]=temp[:,0,1].reshape(self.size_x,self.size_y,int(temp.shape[0]/(self.size_x*self.size_y)),order="C")
         # Mrt_nomag[:,:,:,1,0]=temp[:,1,0].reshape(self.size_x,self.size_y,int(temp.shape[0]/(self.size_x*self.size_y)),order="C")
         # Mrt_nomag[:,:,:,1,1]=temp[:,1,1].reshape(self.size_x,self.size_y,int(temp.shape[0]/(self.size_x*self.size_y)),order="C")
         
         
         
                 
         
            
         # self.Mrt_matrix=Mrt_matrix
         return self.Mrt_matrix
     #this is the main routine called by the XRMS simulation loop.
     
    def get_R(self):
        att2=self.T_fields_linear
        att3=att2*np.conj(att2)
        att=att3.sum(axis=1)
        
        R_array=np.zeros(self.Mrt_matrix.shape,dtype=complex)
        for l in range(att.shape[0]):
            R_array[:,:,l,:,:]=self.Mrt_matrix[:,:,l,:,:]*att[l]
        self.R_array=R_array
        
    def get_M_Beamdirection(self):
        #go from M defined in x, y and z to an M defined as longitudinal (along)
        #the beam, sigma and pi
        
        M=self.M
        theta=self.theta
        
        M_long_in=M[0,:,:,:]*np.cos(theta)+M[2,:,:,:]*np.sin(theta)
        M_long_out=M[0,:,:,:]*np.cos(theta)-M[2,:,:,:]*np.sin(theta)
        M_pi_in=M[0,:,:,:]*np.sin(theta)-M[2,:,:,:]*np.cos(theta)
        M_pi_out=-M[0,:,:,:]*np.sin(theta)-M[2,:,:,:]*np.cos(theta)
        M_trans=M[1,:,:,:]
        return M_long_in, M_long_out, M_pi_in,M_pi_out,M_trans
    def get_U_mag_absorption(self):
        #for the magnetic absorption correction in the 3D case
        #we use a simplified formula to calculate u as defined in equation 6
        #of Stepanov Sinha
        lamda=self.lamda
        f_charge=self.f_charge
        f_Mag=self.f_Mag
        f_Mag2=self.f_Mag2
        r0=self.r0
        na=self.na
               
        multiplier=lamda**2*r0/np.pi
        chi_zero=na*multiplier*f_charge
        u=(chi_zero+np.sin(self.theta)**2)**0.5
        chi_zero=na*multiplier*(f_charge-f_Mag)
        umag=(chi_zero+np.sin(self.theta)**2)**0.5
        chi_zero=na*multiplier*(f_charge-f_Mag2)  
        umag2=(chi_zero+np.sin(self.theta)**2)**0.5        
        return u,umag,umag2
      
        
    def get_Faraday_Parallel(self,specular):
        #this function calculates the differential absorption due to sections of the sample
        #magnetized parallel to the incident beam, and is especially important when 
        #there is an applied field applied.
        
        #For best results, it is advised to use periodic boundary conditions for the micromagnetic 
        #simulations as an input, because the interpolation routine exceeds the boundary of the 
        
        #this code calculates to first order the pertubation in the absorption caused by the magnetization dependent part 
        #of the atomic scattering factor
        
        M_long_in, M_long_out, M_pi_in,M_pi_out,M_trans=self.get_M_Beamdirection()
        theta=self.theta
        def get_abs_adj(XMCD_XMLD,M,in_out):
            nx=self.size_x
            ny=self.size_y
            nz=self.sample.size_z
            
            dx=self.sample.dx
            dz=self.sample.dz
            
            dz_mag=self.sample.dz_mag
            
            ratio=dx/dz
            x=np.linspace(0,nx-1,nx)
            y=np.linspace(0,ny-1,ny)
            z=np.linspace(0,nz-1,nz)
            interp_plus=np.zeros((len(x),len(y),len(z)),dtype=complex)
            interp_minus=np.zeros((len(x),len(y),len(z)),dtype=complex)
            for k in range(len(z)):
                x_shifted_plus=(x+z[k]*ratio/np.tan(theta))%nx
                x_shifted_minus=(x-z[k]*ratio/np.tan(theta))%nx
                pts_plus=[]
                pts_minus=[]
                for i in range (len(x)):
                    for j in range (len(y)):
                        point_plus=[x_shifted_plus[i],y[j]]
                        point_minus=[x_shifted_minus[i],y[j]]
                        pts_plus.append(point_plus)
                        pts_minus.append(point_minus)
                if XMCD_XMLD=="XMCD":
                    temp_plus = RegularGridInterpolator((x, y), M[:,:,k],bounds_error=False, fill_value=0)
                    temp_minus = RegularGridInterpolator((x, y), M[:,:,k],bounds_error=False, fill_value=0) 
                if XMCD_XMLD=="XMLD":
                    temp_plus = RegularGridInterpolator((x, y), M[:,:,k]**2,bounds_error=False, fill_value=0)
                    temp_minus = RegularGridInterpolator((x, y), M[:,:,k]**2,bounds_error=False, fill_value=0) 
                a=np.reshape(np.array(temp_plus(pts_plus)),(len(x),len(y)))
                b=np.reshape(np.array(temp_minus(pts_minus)),(len(x),len(y)))
                interp_plus[:,:,k]=a
                interp_minus[:,:,k]=b
                
            partial_sum_plus=interp_plus-interp_plus
            partial_sum_minus=interp_plus-interp_plus
            
            
            for j in range(interp_plus.shape[2]):
                if j==0:
                    partial_sum_plus[:,:,j]=interp_plus[:,:,j]
                    partial_sum_minus[:,:,j]=interp_minus[:,:,j]
                else:
                    partial_sum_plus[:,:,j]=partial_sum_plus[:,:,j-1]+interp_plus[:,:,j]
                    partial_sum_minus[:,:,j]=partial_sum_minus[:,:,j-1]+interp_minus[:,:,j]
                    
            for k in range(len(z)):
                    
                temp_plus = RegularGridInterpolator((x, y), partial_sum_minus[:,:,k],bounds_error=False, fill_value=np.sum(partial_sum_minus[0,:,k])/M.shape[1])
                temp_minus = RegularGridInterpolator((x, y), partial_sum_plus[:,:,k],bounds_error=False, fill_value=np.sum(partial_sum_minus[0,:,k])/M.shape[1]) 
                a=np.reshape(np.array(temp_plus(pts_plus)),(len(x),len(y)))
                b=np.reshape(np.array(temp_minus(pts_minus)),(len(x),len(y)))
                partial_sum_plus[:,:,k]=b
                partial_sum_minus[:,:,k]=a
            
            
            u,umag,umag2=self.get_U_mag_absorption()   
            x=np.exp(-np.imag(u)*2*np.pi/self.lamda*dz_mag)
            dx=np.exp(-np.imag(umag)*2*np.pi/self.lamda*dz_mag)-x
            dx2=np.exp(-np.imag(umag2)*2*np.pi/self.lamda*dz_mag)-x
            if XMCD_XMLD=="XMCD":
                adj_abs_plus=partial_sum_plus/x*dx+1
                adj_abs_minus=partial_sum_minus/x*dx+1 
            if XMCD_XMLD=="XMLD":
                adj_abs_plus=partial_sum_plus/x*dx2+1
                adj_abs_minus=partial_sum_minus/x*dx2+1 
            if in_out=="in":
                return adj_abs_plus
            if in_out=="out":
                return adj_abs_minus
        adj_abs_plus_XMCD=get_abs_adj(XMCD_XMLD="XMCD",M=M_long_in,in_out="in")
        adj_abs_minus_XMCD=get_abs_adj(XMCD_XMLD="XMCD",M=M_long_out,in_out="out")
        adj_abs_plus_XMLD_pi=get_abs_adj(XMCD_XMLD="XMLD",M=M_pi_in,in_out="in")
        adj_abs_minus_XMLD_pi=get_abs_adj(XMCD_XMLD="XMLD",M=M_pi_out,in_out="out")
        adj_abs_plus_XMLD_trans=get_abs_adj(XMCD_XMLD="XMLD",M=M_trans,in_out="in")
        adj_abs_minus_XMLD_trans=get_abs_adj(XMCD_XMLD="XMLD",M=M_trans,in_out="out")
        #absorption factor        
        return adj_abs_plus_XMCD,adj_abs_minus_XMCD,adj_abs_plus_XMLD_pi,adj_abs_minus_XMLD_pi,adj_abs_plus_XMLD_trans,adj_abs_minus_XMLD_trans
        
        
        
    def R_array_2_Diffraction(self):
        
        R=self.sample.R_interp
        z=self.sample.z_interp
        
        
        Fourier2d=np.fft.fftshift(np.fft.fftn(R, axes=(0, 1)),axes=(0,1)) 
        
        nn=R.shape[1]
        delta_z_Fourier=sample.unit_cell/2
        n_average=self.n_average
        z_index=(np.array(z-z[np.int64(np.round(len(z)/2))])/delta_z_Fourier*np.real(n_average))        
        delta_qx=4*np.pi/nn/self.sample.dx
        delta_qz=4*np.pi/max(z_index)/delta_z_Fourier
        pix_z=np.linspace(-nn/2,nn/2-1,nn)
        temp=np.ones((1,nn))
        kz=2*np.pi/self.lamda*np.sin(self.theta)
        q_z=np.outer(pix_z,temp)*delta_qx*np.cos(self.theta)/np.sin(self.theta)+kz*2
        #q_z value corresponding to each q_x value
        #q_z=q_z-q_z+kz
        qz_index=(q_z/delta_qz)
        DFT_zeros=np.zeros((nn,nn,len(z_index)),dtype=complex)
        @njit
        def get_DFT(nn,z_index,qz_index,DFT_zeros):
            DFT=DFT_zeros
            for l in range (len(z_index)):
                #print(l)
                DFT[:,:,l]=np.exp(-2*np.pi*complex(0,1)/max(z_index)*qz_index*z_index[l])
            return DFT
        DFT=get_DFT(nn,z_index,qz_index,DFT_zeros) 
                   #this line is to evaluate the 3D Fourier transform only over the Ewald Sphere
        self.XRMS_Pure=[[np.sum(DFT*Fourier2d[:,:,:,0,0],2),np.sum(DFT*Fourier2d[:,:,:,0,1],2)],\
                        [np.sum(DFT*Fourier2d[:,:,:,1,0],2),np.sum(DFT*Fourier2d[:,:,:,1,1],2)]]
        self.XRMS=self.XRMS_Pure
        
    def export_intensity(self,Incident):
        In=Incident
        XRMS=self.XRMS
        self.intensity=(In[0]*XRMS[0][0]+In[1]*XRMS[0][1])*np.conj(In[0]*XRMS[0][0]+In[1]*XRMS[0][1])+\
        (In[0]*XRMS[1][0]+In[1]*XRMS[1][1])*np.conj(In[0]*XRMS[1][0]+In[1]*XRMS[1][1])
        return self.intensity
    def display_intensity(self,Incident):
        diffraction=self.export_intensity(Incident)
        
        diffraction[:,100]=0
        #axs[0].plot(np.sum(diffraction[80:120,80:120],axis=0)) 
        
        im=(np.real(diffraction[90:110,90:110]))
        
        return im
    


output_array=np.zeros((200,200,501))
output_array_conj=np.zeros((200,200,501))
fig,ax = plt.subplots(1,1)

#fig.suptitle('XRMS output')
plt.ylabel("pixel_long")
plt.xlabel("pixel_trans")
count=0
ims = []  
angles=np.linspace(10,40,61)
for theta_deg in angles:
    print(theta_deg)
    sample=LSMO()
    Incident=sample.Incident
    Incident2=np.array([[complex(1,0)],[complex(0,-1)]])
    theta=theta_deg*np.pi/180
    specular=Stepanov_Specular(sample,theta,energy=640,magnetic_diff='off')
    specular.Chi_define()
    specular.get_A_S_matrix()
    specular.XM_define_Big()
    specular.Matrix_stack()
    
    XRMS=XRMS_Simulate(sample,specular,theta,energy=640)
    XRMS.Chi_define()
    XRMS.get_A_S_matrix()
    output=XRMS.XM_define_Small()
    XRMS.get_R()
    XRMS.get_Faraday_Parallel(specular)
    sample.interpolate_nearest_neighbour(XRMS.R_array)
    XRMS.R_array_2_Diffraction()
    #fig=plt.figure()
    
        
    #ax.set_title(f'XRMS output_{theta_deg}')
    #ims.append([im])

    
    output_array[:,:,count]=XRMS.export_intensity(Incident)
    output_array_conj[:,:,count]=XRMS.export_intensity((Incident2))
    count=count+1
def update(j):
    theta_deg=angles
    diffraction=output_array[:,:,j]-output_array_conj[:,:,j]
    diffraction[:,100]=0
    #axs[0].plot(np.sum(diffraction[80:120,80:120],axis=0)) 
    
    im=(np.real(diffraction[90:110,90:110]))
    plt.title(f'XRMS output_{theta_deg[j]}')
    plt.imshow(im)
ani = animation.FuncAnimation(fig,update,frames=len(angles), interval=200, blit=False,repeat_delay=1000)
ani.save("XRMS.gif")
#Bloch_correction=XRMS.get_Faraday_Parallel(specular)
#output_array_conj2=np.resize(output_array_conj,(40000,501))

#output_array2=np.resize(output_array,(40000,501))
#np.savetxt('sigma'+'deep_LSMO'+'.csv',output_array2, delimiter=',')
#np.savetxt('pi'+'deep_LSMO'+'.csv',output_array_conj2, delimiter=',')
    


         
