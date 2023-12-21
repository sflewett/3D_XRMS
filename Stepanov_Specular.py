# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 13:14:38 2023

@author: sflewett
"""
import numpy as np
from numba import jit,njit
from Sample_Class_test import XRMS_Sample
import time

class Stepanov_Specular():
    
    def __init__(self, sample,theta,energy,magnetic_diff='on'):
        
        c=3e8
        f=energy*1.6e-19/6.626e-34
        self.lamda=c/f
        self.theta=theta
        self.r0=2.82e-15
        f_Charge=sample.f_Charge
        f_Mag=sample.f_Mag
        na=sample.na
        M=sample.M
        self.f_Charge=f_Charge[0,0,:]
        self.f_Mag=f_Mag[0,0,:]
        self.f_Mag2=sample.f_Mag2[0,0,:]
        self.na=na[0,0,:]
        self.M=(sample.M[:,20,20,:])
        #taking only one column for the specular calculation
        self.z=sample.z
        d=np.diff(self.z)
        d2=np.zeros((len(d)+1))
        d2[1:len(d)+1]=d
        d2[0]=d[1];
        self.d=d2
        self.Incident=sample.Incident#polarization basis of the incident light (sigma, pi)
        self.sigma_roughness=sample.sigma_roughness
        self.magnetic_diff=magnetic_diff
    def Chi_define(self):
        
        #M is set up as [m_vector,xpos,ypos,zpos]
        #Chi as defined in the Stepanov Sinha Paper
        #X:longitudinal
        #Y: Transverse
        #Z: polar
        #na es la densidad de átomos
        lamda=self.lamda
        f_charge=self.f_Charge
        f_Mag=self.f_Mag
        f_Mag2=self.f_Mag2
        r0=self.r0
        na=self.na
        M=self.M
        multiplier=lamda**2*r0/np.pi
        chi_zero=np.zeros((len(self.z)),dtype=complex)
        for j in range(len(self.z)):
            chi_zero[j]=na[j]*multiplier*f_charge[j]
        if self.magnetic_diff=='on':
            B=na*multiplier*f_Mag
        if self.magnetic_diff=='off':
            B=na*multiplier*0.0001
        C=na*multiplier*f_Mag2#This needs to be set in the case that a quadratic term is present
       #And should be set in the initialization routine as a global parameter
        chi=np.zeros((3, 3),dtype=complex)
       #THIS PART NEEDS TO BE BETTER COMMENTED TO REFERENCE TO THE CORRECT PAPERS (STEPANOV, HANNON ETC)
        
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
        self.chi2=-complex(0,1)*B*temp+C*MM
        self.chi=chi
        self.chi_zero=chi_zero*ones
        
        
        
    def get_U(self):
        #Equations 25 to 30 of Stepanov-Sinha
        M=self.M
        chi=self.chi
        chi_zero=self.chi_zero
        theta=self.theta
        
        nx=np.array(np.cos(theta),dtype=complex);
        gamma=np.array(np.sin(theta),dtype=complex);
        M_tot=(M**2).sum(axis=0)
        M_tot2=M_tot
                
        mask1=M_tot!=0
        mask2=M_tot==0
        #import quartic_solver_analytic
        
        def Uloop(Q1,Q2,Q3,Q4,Q5,output_arrays):
            u1=output_arrays[0]
            u2=output_arrays[1]
            u3=output_arrays[2]
            u4=output_arrays[3]
            for l in range(len(self.z)):
                q=np.array([Q1[l],Q2[l],Q3[l],Q4[l],Q5[l]])
                temp1,temp2,temp3,temp4=np.roots(q)                                        
                u1[l],u2[l],u3[l],u4[l]=temp1,temp2,temp3,temp4
            return u1,u2,u3,u4
        
        
        def Umag(chi,nx,gamma):
            u1=np.zeros((len(self.z)),dtype=np.float64)
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
            u1,u2,u3,u4=Uloop(Q1,Q2,Q3,Q4,Q5,output_arrays)
            u=np.array([u1,u2,u3,u4],dtype=complex)
            #here we literally copy from equations 25-30 of Stepanov Sinha
            imag=u*complex(0,1)
            u_sorted=np.sort(imag,axis=0)
            u_sorted=u_sorted/complex(0,1)
            temp=np.zeros((u_sorted.shape),dtype=complex)
            temp[0:2,:]=u_sorted[0:2,:]
            temp[2,:]=u_sorted[2,:]
            temp[3,:]=u_sorted[3,:]
            u_sorted=temp
            u=u_sorted
            D=(chi[0,2]+u*nx)*(chi[2,0]+u*nx)-(1-u**2+chi[0,0])*(gamma**2+chi[2,2])
            Px=(chi[0,1]*(gamma**2+chi[2,2])-chi[2,1]*(chi[0,2]+u*nx))/D
            #equations 33 and 35 of Stepanov-Sinha
            # we calculate the eigenvector here as a means for sorting the order of the eigenvalues, and therefore the filling of the matrices
            #not including this step produces sudden shifts in the order of the eigenwaves predicted by the quartic solver
            u_i=u[0:2,:]
            u_r=u[2:4,:]
            
            Px_i=np.imag(Px[0:2,:])
            Px_r=np.imag(Px[2:4,:])
            
            sort_ind_i=np.argsort(Px_i,axis=0)
            u_i_sorted=np.take_along_axis(u_i,sort_ind_i,axis=0)
            
            sort_ind_r=np.argsort(Px_r,axis=0)
            u_r_sorted=np.take_along_axis(u_r,sort_ind_r,axis=0)
            
            u[0:2,:]=u_i_sorted
            u[2,:]=u_r_sorted[1,:]
            u[3,:]=u_r_sorted[0,:]
            # the sorting is done such that there are no sudden phase jumps in Px and Pz
            return u
        def Unonmag(chi_zero,nx,gamma):# non magnetic case
            #Equation 6 of Stepanov Sinha
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
        #Equation 15 of Stepanov Sinha in the non-magnetic case, and equation 36 in the magnetic case
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
             self.get_U()
             u=self.U
             epsilon=1+chi_zero
             ones=np.ones(M_tot.shape)
             zeros=np.zeros(M_tot.shape)
             Matrix=[[ones,zeros,ones,zeros],\
                    [zeros,epsilon**0.5,zeros,epsilon**0.5],\
                    [u[0,...],zeros,u[1,...],zeros],\
                    [zeros,u[0,...]/epsilon**0.5,zeros,u[1,...]/epsilon**0.5]]
             zeros=u[0,...]-u[0,...]    
             F=[[np.exp(-complex(0,1)*u[0,...]*2*np.pi/self.lamda*self.d),zeros,zeros,zeros],\
                      [zeros,np.exp(-complex(0,1)*u[0,...]*2*np.pi/self.lamda*self.d),zeros,zeros],\
                      [zeros,zeros,np.exp(-complex(0,1)*u[1,...]*2*np.pi/self.lamda*self.d),zeros],\
                      [zeros,zeros,zeros,np.exp(-complex(0,1)*u[1,...]*2*np.pi/self.lamda*self.d)]]
             P_dims=[4]+list(M_tot.shape) 
             Px=np.zeros((P_dims))
             Pz=Px
             #for the non-magnetic case, the eigenvectors, whose x and z components are labelled Px and Pz
             #do not have a meaning expressed in terms of Py, as is the case for the magnetic case
             return np.array(Matrix,dtype=complex),np.array(F,dtype=complex),Px,Pz
         def Small_matrix_mag(chi,chi_zero,theta,nx,gamma):
             self.get_U()
             u=self.U
             D=(chi[0,2,...]+u*nx)*(chi[2,0,...]+u*nx)-(1-u**2+chi[0,0,...])*(gamma**2+chi[2,2,...])
             Px=(chi[0,1,...]*(gamma**2+chi[2,2,...])-chi[2,1,...]*(chi[0,2,...]+u*nx))/D
             Pz=(chi[2,1,...]*(1-u**2+chi[0,0,...])-chi[0,1,...]*(chi[2,0,...]+u*nx))/D
             #equations 33, 35 and 35 of Stepanov Sinha
             v=u*Px-nx*Pz
             w=Px
             #equations 37 and 38 of Stepanov Sinha
             ones=np.ones(M_tot.shape,dtype=complex)
             zeros=u[0,...]-u[0,...]
             Matrix=[[ones,ones,ones,ones],\
                     [v[0,...],v[1,...],v[2,...],v[3,...]],\
                     [u[0,...],u[1,...],u[2,...],u[3,...]],\
                     [w[0,...],w[1,...],w[2,...],w[3,...]]]                     
                
             F=[[np.exp(-complex(0,1)*u[0,...]*2*np.pi/self.lamda*self.d),zeros,zeros,zeros],\
                     [zeros,np.exp(-complex(0,1)*u[1,...]*2*np.pi/self.lamda*self.d),zeros,zeros],\
                     [zeros,zeros,np.exp(-complex(0,1)*u[2,...]*2*np.pi/self.lamda*self.d),zeros],\
                     [zeros,zeros,zeros,np.exp(-complex(0,1)*u[3,...]*2*np.pi/self.lamda*self.d)]]
             return np.array(Matrix,dtype=complex),np.array(F,dtype=complex),Px,Pz
         
         A_S_matrix_dims=[4,4]+list(M_tot.shape) 
         #the naming comes from the fact that some resources name this medium boundary matrix A, whereas Stepanov Sinha uses S
         A_S_matrix=np.zeros((A_S_matrix_dims),dtype=complex)
         F_matrix=np.zeros((A_S_matrix_dims),dtype=complex)
         P_dims=[4]+list(M_tot.shape) 
         self.Px=np.zeros((P_dims),dtype=complex)
         self.Pz=np.zeros((P_dims),dtype=complex)
         #initializing arrays
         temp1,temp2,temp3,temp4=Small_matrix_mag(chi,chi_zero,theta,nx,gamma)
         A_S_matrix[:,:,mask1],F_matrix[:,:,mask1],self.Px[:,mask1],self.Pz[:,mask1]=temp1[:,:,mask1],temp2[:,:,mask1],temp3[:,mask1],temp4[:,mask1]
         temp1,temp2,temp3,temp4=Small_matrix_nomag(chi,chi_zero,theta)
         A_S_matrix[:,:,mask2],F_matrix[:,:,mask2],self.Px[:,mask2],self.Pz[:,mask2]=temp1[:,:,mask2],temp2[:,:,mask2],temp3[:,mask2],temp4[:,mask2]
                  
         A_S_matrix2=np.zeros((A_S_matrix.shape[0],A_S_matrix.shape[1],A_S_matrix.shape[2]+1),dtype=complex)
         F_matrix2=np.zeros((A_S_matrix.shape[0],A_S_matrix.shape[1],A_S_matrix.shape[2]+1),dtype=complex)
         Matrix=np.array([[1,0,1,0],\
                [0,1,0,1],\
                [gamma,0,-gamma,0],\
                [0,gamma,0,-gamma]])
         F_vacuum=np.array([[1,0,0,0],\
                [0,1,0,0],\
                [0,0,1,0],\
                [0,0,0,1]])
         #this is the vacuum matrix from the LHS of equation 15
         A_S_matrix2[...,1:A_S_matrix.shape[-1]+1]=A_S_matrix
         A_S_matrix2[...,0]=Matrix
         F_matrix2[...,1:A_S_matrix.shape[-1]+1]=F_matrix
         F_matrix2[...,0]=F_vacuum
         
         self.AS_Matrix=A_S_matrix2
         self.F_Matrix=F_matrix2
         Px2=np.zeros((self.Px.shape[0],self.Px.shape[1]+1),dtype=complex)
         Pz2=np.zeros((self.Px.shape[0],self.Px.shape[1]+1),dtype=complex)
         Px2[...,1:self.Px.shape[-1]+1]=self.Px
         Pz2[...,1:self.Pz.shape[-1]+1]=self.Pz
         self.Px=Px2
         self.Pz=Pz2
         
             
    def XM_define_Big(self):
         #this method has the purpose of going through equations 55 through 62 of Stepanov Sinha, using the same notation.
         #Unlike for the 3D case, there is no need to perform changes of bases here
         A_S_Matrix=self.AS_Matrix
         F2=self.F_Matrix
         AS1=A_S_Matrix[...,0:A_S_Matrix.shape[-1]-1]
         AS2=A_S_Matrix[...,1:A_S_Matrix.shape[-1]]
         F2=F2[...,1:A_S_Matrix.shape[-1]]
         permutation=(2,0,1)
         AS1=np.transpose(AS1, permutation)
         AS2=np.transpose(AS2, permutation)
         F2=np.transpose(F2, permutation)
         X=np.matmul(np.linalg.inv(AS1),AS2)
         self.X=X
         self.F2_mas=F2[...,0:2,0:2]
         self.F2_menos=F2[...,2:4,2:4]
         self.Xtt=X[...,0:2,0:2]
         self.Xtr=X[...,0:2,2:4]
         self.Xrt=X[...,2:4,0:2]
         self.Xrr=X[...,2:4,2:4]
         self.Mtt=np.linalg.inv(self.F2_mas)@np.linalg.inv(self.Xtt)
         self.Mtr=-self.Mtt@self.Xtr@self.F2_menos
         self.Mrt=self.Xrt@np.linalg.inv(self.Xtt)
         self.Mrr=(self.Xrr-self.Mrt@self.Xtr)@self.F2_menos
             
    def Matrix_stack(self):
        #performing the recursive matrix stack operation for calculating the specular reflection. 
        #Equations 63 through 70 of Stepanov Sinha
        self.count=0
        kz=2*np.pi/self.lamda*np.sin(self.theta)
        roughness_reduction=np.exp(-kz**2*self.sigma_roughness**2)
        #Empezando en la superficie
        #Mtt,Mtr,Mrt,Mrr,Px,Pz,u=self.Mtt,self.Mtr,self.Mrt,self.Mrr,self.Px,self.Pz,self.U
        #Agregando la parte de la rugosidad de Nevot-Croce, Ecuación 5,7 de Lee (1) (Aproximación de Born)
        T0=(self.Incident)
        M_list_tt=[]
        M_list_tr=[]
        M_list_rt=[]
        M_list_rr=[]
        W_list_tt=[]
        W_list_tr=[]
        W_list_rt=[]
        W_list_rr=[]
        R_fields=[]
        T_fields=[]
        tempzeros=np.array([[0],[0]])
        M_list_tt.append(self.Mtt[0,:,:])
        M_list_tr.append(self.Mtr[0,:,:]*roughness_reduction)
        M_list_rt.append(self.Mrt[0,:,:]*roughness_reduction)#Aplicando la rugosidad solamente a las partes relacionados a la reflexión
        M_list_rr.append(self.Mrr[0,:,:])
        W_list_tt.append(self.Mtt[0,:,:])
        W_list_tr.append(self.Mtr[0,:,:]*roughness_reduction)
        W_list_rt.append(self.Mrt[0,:,:]*roughness_reduction)
        W_list_rr.append(self.Mrr[0,:,:])
        R_fields.append(tempzeros)
        T_fields.append(tempzeros)
        for i in range(1,len(self.z)):
            #print(params[i])
            
            #Selección de parametros elementales
            Mtt,Mtr,Mrt,Mrr=self.Mtt[i,:,:],self.Mtr[i,:,:],self.Mrt[i,:,:],self.Mrr[i,:,:]
            M_list_tt.append(Mtt)
            M_list_tr.append(Mtr*roughness_reduction)
            M_list_rt.append(Mrt*roughness_reduction)
            M_list_rr.append(Mrr)
            II=np.array([[1,0],[0,1]])
            A=M_list_tt[i]@np.linalg.inv(II-W_list_tr[i-1]@M_list_rt[i])
            B=W_list_rr[i-1]@np.linalg.inv(II-M_list_rt[i]@W_list_tr[i-1])
            
            ##OJO AQUÏ Papers tienen definiciones diferentes
            Wtt=A@W_list_tt[i-1]
            Wtr=M_list_tr[i]+A@W_list_tr[i-1]@M_list_rr[i]
            Wrt=W_list_rt[i-1]+B@M_list_rt[i]@W_list_tt[i-1]
            Wrr=B@M_list_rr[i]
            #Ecuaciones 66 y 67 de Stepanov Sinha
            W_list_tt.append(Wtt)
            W_list_tr.append(Wtr)
            W_list_rt.append(Wrt)
            W_list_rr.append(Wrr)
            R_fields.append(tempzeros)#Inicializando las listas
            T_fields.append(tempzeros)
            self.count=self.count+1
        for i in range(len(self.z)-2,-1,-1):
            R_fields[i]=np.linalg.inv(II-M_list_rt[i+1]@W_list_tr[i])@(M_list_rr[i+1]@R_fields[i+1]+M_list_rt[i+1]@W_list_tt[i]@T0)
            T_fields[i]=W_list_tt[i]@T0+W_list_tr[i]@R_fields[i]
       #Ecuación 70 de Stepanov Sinha
        self.T_fields=np.array(T_fields)
        self.R_fields=np.array(R_fields)
        #Fields within the sample in their eigen-bases for each particular layer
        self.specular_output=np.array(W_list_rt[-1])
        R=self.specular_output@T0
        self.I_output=np.abs(R[0])**2+np.abs(R[1])**2
        #just changed the code to output the matrix of overall reflection coefficients
        
        Px2=self.Px[...,1:self.Px.shape[-1]]
        
        Pz2=self.Pz[...,1:self.Pz.shape[-1]]
        Basischange=self.Xrt-self.Xrt
        ones=np.ones(self.T_fields.shape[0],dtype=complex)
        efields=np.zeros((Basischange.shape[0],4,3),dtype=complex)
        for j in range (4):
            efields[...,j,0]=Px2[j,...]
            efields[...,j,1]=ones
            efields[...,j,2]=Pz2[j,...]
            
        
        Basischange[...,0,0]=ones
        Basischange[...,0,1]=ones
        Basischange[...,1,0]=Px2[0,...]*np.sin(self.theta)+Pz2[0,...]*np.cos(self.theta)
        Basischange[...,1,1]=Px2[1,...]*np.sin(self.theta)+Pz2[1,...]*np.cos(self.theta)
        #puts the polarization into the s-p basis
        Basischange2=Basischange
        Basischange2[...,1,0]=Px2[2,...]*np.sin(self.theta)+Pz2[2,...]*np.cos(self.theta)
        Basischange2[...,1,1]=Px2[3,...]*np.sin(self.theta)+Pz2[3,...]*np.cos(self.theta)
        #basis change matrix for converting the in-sample fields to the sigma-pi basis
        for l in range(Basischange.shape[0]):
            if np.linalg.det(Basischange[l,:,:])==0:
                Basischange[l,:,:]=np.identity(2,dtype=complex)
                Basischange2[l,:,:]=np.identity(2,dtype=complex)
                for j in range (4):
                    if j%2==0:
                        efields[l,j,0]=0
                        efields[l,j,1]=1
                        efields[l,j,2]=0
                    else:
                        efields[l,j,0]=np.sin(self.theta)
                        efields[l,j,1]=0
                        efields[l,j,2]=np.cos(self.theta)
        self.efields=efields                
        self.Basischange=np.array(Basischange)
        self.Basischange2=np.array(Basischange2)
        T_fields_linear=np.zeros(self.T_fields.shape,dtype=complex)
        R_fields_linear=np.zeros(self.R_fields.shape,dtype=complex)
        for l in range(Basischange.shape[0]):
            T_fields_linear[l,:,:]=(self.Basischange[l,:,:])@self.T_fields[l,:]
            R_fields_linear[l,:,:]=(self.Basischange2[l,:,:])@self.R_fields[l,:]    
        self.T_fields_linear=np.array(T_fields_linear)
        self.R_fields_linear=np.array(R_fields_linear)
        #outputting the fields in the sigma-pi basis
        