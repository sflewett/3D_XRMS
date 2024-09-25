# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 09:40:32 2023

@author: sflewett
"""
import numpy as np
import numpy.ma as ma
from operator import and_
from scipy.interpolate import RegularGridInterpolator
from numba import njit

class XRMS_Simulate():
    def __init__(self, sample,theta,energy,full_column="full",column=[0,0]):
        #input is a sample class "sample", an angle of incidence "theta" in radians, a photon energy "energy" in eV, 
        #a string "full" or "column" to decide to evaluate over the whole array, or a single column within
        #and the indices of this column given by the 2-vector column. 
        c=3e8
        f=energy*1.6e-19/6.626e-34
        self.lamda=c/f
        self.theta=theta
        self.f_charge=sample.f_Charge
        self.f_charge_maglayer_scalar=sample.f_Charge_maglayer_scalar
        self.f_Mag_scalar=sample.f_Mag_scalar
        self.na_scalar=sample.na_scalar
        self.f_Mag=sample.f_Mag
        ##WE NEED TO LOOK INTO THIS ISSUE OF THE MAGNETIC SCATTERING FACTORS!!##
        self.f_Mag2=sample.f_Mag2
        self.full_column=full_column
        self.r0=2.82e-15
        self.na=sample.na
        self.M=sample.M
        self.M_tot=(self.M**2).sum(axis=0)
        self.My=self.M[1,:]
        self.TMOKE_Threshold=sample.TMOKE_Threshold
        self.mask1=self.M_tot==0
        self.mask1[np.abs(self.My)>self.TMOKE_Threshold]=True
        self.mask2=self.M_tot!=0
        self.mask3=np.abs(self.My)<self.TMOKE_Threshold
        
        if full_column=="column":
            self.M=sample.M[:,column[0]:column[0]+1,column[1]:column[1]+1,:]
            self.M_tot=(self.M**2).sum(axis=0)
            self.My=self.M[1,:]
            self.mask1=self.M_tot==0
            self.mask1[np.abs(self.My)>self.TMOKE_Threshold]=True
            self.mask2=self.M_tot!=0
            self.mask3=np.abs(self.My)<self.TMOKE_Threshold
            self.f_charge=sample.f_Charge[column[0]:column[0]+1,column[1]:column[1]+1,:]
            self.f_Mag=sample.f_Mag[column[0]:column[0]+1,column[1]:column[1]+1,:]
            self.f_Mag2=sample.f_Mag2[column[0]:column[0]+1,column[1]:column[1]+1,:]
            self.na=sample.na[column[0]:column[0]+1,column[1]:column[1]+1,:]
        self.z=sample.z
        d=np.diff(self.z)
        d2=np.zeros((len(d)+1))
        d2[1:len(d)+1]=d
        d2[0]=d[1];
        self.d=d2
        #we have to set masks for the different calculation regimes
        #1. Magnetic normal
        #2. Non-magnetic (reduces to the standard Fresnel Formulae)
        #3. Magnetic TMOKE
        self.size_x=sample.size_x
        self.size_y=sample.size_y
        if full_column=="column":
            self.size_x=sample.size_x
            self.size_y=sample.size_y
        #self.T_fields_linear=specular.T_fields_linear
        self.sample=sample
        self.factor=self.na*self.r0*self.lamda**2/2/np.pi
        self.factor=np.sum(self.factor)/np.product(self.factor.shape)
        f_average=np.sum(self.f_charge)/np.product(self.f_charge.shape)
        self.n_average=1-self.factor*f_average
        self.Incident=sample.Incident#polarization basis of the incident light (sigma, pi)
        self.sigma_roughness=sample.sigma_roughness
        
        
    def mask_expand(self,mask,array_dimens,before_after="before"):
        #takes the mask and expands its dimensionality to the new mask dimensions:
        #if the new mask is to have dimensions [array_dimens, old mask dimensions],
        #then select "before",otherwise select "after"
            def expand_one_dimension(mask,dimension,before_after):
                mask_expanded=np.array([mask for i in range(dimension)])
                if before_after=="after":
                    dims=len(mask.shape)
                    permutation=[]
                    for i in range(dims):
                        permutation.append(i+1)
                    permutation.append(0)
                    mask_expanded=np.transpose(mask_expanded, permutation)
                return mask_expanded
            
            if before_after=="after":
                array_dimens.reverse()
            for j in range(len(array_dimens)):
                mask=expand_one_dimension(mask,array_dimens[j],before_after)
            
            return mask
                    
    def Chi_define(self):
        #M is set up as [m_vector,xpos,ypos,zpos]
        #Chi as defined in the Stepanov Sinha Paper
        #X:longitudinal
        #Y: Transverse
        #Z: polar
        #na is the number of atoms per square meter.
        #the above parameters are all set in the Sample class
        
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
        
        norm=np.sqrt(m1**2+m2**2+m3**2+1e-9)
        
        m1=m1/norm
        m2=m2/norm
        m3=m3/norm
        
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
        self.chi_mask1=self.mask_expand(self.mask1,[3,3],before_after="before")
        self.chi_mask2=self.mask_expand(self.mask2,[3,3],before_after="before")
        self.chi_mask3=self.mask_expand(self.mask3,[3,3],before_after="before")
        self.chi_masked1=ma.masked_array(chi,self.chi_mask1)
        self.chi_masked2=ma.masked_array(chi,self.chi_mask2)
        self.chi_masked3=ma.masked_array(chi,self.chi_mask3)
        self.chi_zero_masked2=ma.masked_array(chi_zero,self.mask2)
        
    def get_U(self):
        #Equations 25 to 30 of Stepanov-Sinha
        theta=self.theta
        
        nx=np.array(np.cos(theta),dtype=complex);
        gamma=np.array(np.sin(theta),dtype=complex);
        
        dim1=self.M_tot.shape[0]
        dim2=self.M_tot.shape[1]
        dim3=self.M_tot.shape[2]
        dim=np.array([dim1,dim2,dim3],dtype=np.int64)
                 
        @njit
        #Without njit, this is by far the slowest part of the entire code
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
            u1,u2,u3,u4=Uloop(self.M_tot,Q1,Q2,Q3,Q4,Q5,output_arrays)
            u=np.array([u1,u2,u3,u4],dtype=complex)
            #here we literally copy from equations 25-30 of Stepanov Sinha
            imag=u*complex(0,1)
            u_sorted=np.sort(imag,axis=0)
            u_sorted=u_sorted/complex(0,1)
            temp=np.zeros((u_sorted.shape),dtype=complex)
            temp[0:2,:,:,:]=u_sorted[0:2,:,:,:]
            temp[2,:,:,:]=u_sorted[2,:,:,:]
            temp[3,:,:,:]=u_sorted[3,:,:,:]
            u_sorted=temp
            u=u_sorted
            
            #first filtering the output of the np.roots solver in terms of descending order of the imaginary part
            
            D=(chi[0,2,...]+u*nx)*(chi[2,0,...]+u*nx)-(1-u**2+chi[0,0,...])*(gamma**2+chi[2,2,...])
            Px=(chi[0,1,...]*(gamma**2+chi[2,2,...])-chi[2,1,...]*(chi[0,2,...]+u*nx))/D
            #equations 33 and 35 of Stepanov-Sinha
            # we calculate the eigenvector here as a means for sorting the order of the eigenvalues, and therefore the filling of the matrices
            #not including this step produces sudden shifts in the order of the eigenwaves predicted by the quartic solver
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
            #Equation 6 of Stepanov Sinha
            temp=chi_zero.shape
            temp2=chi_zero.flatten()
            u1=(temp2+gamma**2)**0.5
            u1=u1
            u1_i=np.imag(u1)            
            u1[u1_i<0]=-u1[u1_i<0]
            u1=u1.reshape(temp)
            
            u2=-u1
            u3=u2-u2#setting these to zero
            u4=u3
            u=np.array([u1,u2,u3,u4],dtype=complex)
            
            return u
        def U_TMOKE(chi,nx,gamma):# non magnetic case
            #Equations 39-44 of Stepanov Sinha
            u1=np.zeros((dim),dtype=np.float64)
            u2=u1
            u3=u1
            u4=u1
            delta=chi[0,2,...]**2*(1+chi[0,0,...])
            u1=(gamma**2+chi[1,1,...])**0.5
            temp=chi[0,2,...].shape
            u1a=u1.flatten()
            u1_i=np.imag(u1a)            
            u1a[u1_i<0]=-u1a[u1_i<0]
            u1=u1a.reshape(temp)        
            
            u3=-u1
            u2=(gamma**2+chi[2,2,...]+delta)**0.5
            u2a=u2.flatten()
            u2_i=np.imag(u1a)            
            u2a[u2_i<0]=-u2a[u1_i<0]
            u2=u2a.reshape(temp)       
            
            u4=-u2
            u=np.array([u1,u2,u3,u4],dtype=complex)
            
            return u       
        
        self.UMask1=self.mask_expand(self.mask1,[4],before_after="before")
        self.UMask2=self.mask_expand(self.mask2,[4],before_after="before")
        self.UMask3=self.mask_expand(self.mask3,[4],before_after="before")
        U1=np.array(Umag(self.chi,nx,gamma))
        U2=np.array(Unonmag(self.chi_zero_masked2,nx,gamma))
        U3=np.array(U_TMOKE(self.chi_masked3,nx,gamma))
        self.U1_masked=ma.masked_array(U1,self.UMask1)
        self.U2_masked=ma.masked_array(U2,self.UMask2)
        self.U3_masked=ma.masked_array(U3,self.UMask3)
        a=self.U1_masked
        a.data[a.mask==True]=0
        b=self.U2_masked
        b.data[b.mask==True]=0
        c=self.U3_masked
        c.data[c.mask==True]=0
        self.U=np.ma.array(a.data+b.data+c.data,mask=list(map(and_,map(and_,a.mask,b.mask),c.mask))).data
        #Ecuaciones 25-30 de Stepanov Sinha
    
    def get_A_S_matrix(self):#medium boundary matrix
        #Equation 15 of Stepanov Sinha in the non-magnetic case, and equation 36 in the magnetic case, and equation 42 in the TMOKE case
         
         chi_zero=self.chi_zero
         theta=self.theta
         nx=np.array(np.cos(theta),dtype=complex);
         gamma=np.array(np.sin(theta),dtype=complex);
         M_tot=self.M_tot
         def Matrix_nomag(chi,chi_zero,theta):
             u=self.U2_masked
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
             Px=np.zeros((P_dims),dtype=complex)
             Pz=Px
             #for the non-magnetic case, the eigenvectors, whose x and z components are labelled Px and Pz
             #do not have a meaning expressed in terms of Py, as is the case for the magnetic case
             return np.array(Matrix,dtype=complex),np.array(F,dtype=complex),Px,Pz
         def Matrix_mag(chi,chi_zero,theta,nx,gamma):
             u=self.U1_masked
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
         
         def Matrix_TMOKE(chi,nx,gamma):
             u=self.U3_masked
             delta=chi[0,2,...]**2*(1+chi[0,0,...])
             Rx=-(u*nx+chi[0,2,...])/(nx**2+delta)
             v=u*Rx-nx
             w=Rx
             zeros=np.zeros(M_tot.shape)
             ones=np.ones(M_tot.shape,dtype=complex)
             Matrix=[[ones,zeros,ones,zeros],\
                     [zeros,v[1,...],zeros,v[3,...]],\
                     [u[0,...],zeros,u[2,...],zeros],\
                     [zeros,w[1,...],zeros,w[3,...]]]
             F=[[np.exp(-complex(0,1)*u[0,...]*2*np.pi/self.lamda*self.d),zeros,zeros,zeros],\
                     [zeros,np.exp(-complex(0,1)*u[1,...]*2*np.pi/self.lamda*self.d),zeros,zeros],\
                     [zeros,zeros,np.exp(-complex(0,1)*u[2,...]*2*np.pi/self.lamda*self.d),zeros],\
                     [zeros,zeros,zeros,np.exp(-complex(0,1)*u[3,...]*2*np.pi/self.lamda*self.d)]]
             return np.array(Matrix,dtype=complex),np.array(F,dtype=complex),Rx
         
         
         self.get_U()
         A_S_Mask1=self.mask_expand(self.mask1,[4,4],before_after="before")
         A_S_Mask2=self.mask_expand(self.mask2,[4,4],before_after="before")
         A_S_Mask3=self.mask_expand(self.mask3,[4,4],before_after="before")
         
         temp1,temp2,temp3,temp4=Matrix_mag(self.chi_masked1,chi_zero,theta,nx,gamma)
         A_S_matrix1,F_matrix1,self.Px1,self.Pz1=temp1,temp2,temp3,temp4
         self.A_S_Matrix1_masked=ma.masked_array(A_S_matrix1,A_S_Mask1)
         self.F_Matrix1_masked=ma.masked_array(F_matrix1,A_S_Mask1)
         self.Px1_masked=ma.masked_array(self.Px1,self.U1_masked)
         self.Pz1_masked=ma.masked_array(self.Pz1,self.U1_masked)   
                            
         temp1,temp2,temp3,temp4=Matrix_nomag(self.chi_masked2,chi_zero,theta)
         A_S_matrix2,F_matrix2,self.Px2,self.Pz2=temp1,temp2,temp3,temp4
         self.F_Matrix2_masked=ma.masked_array(F_matrix2,A_S_Mask2)
         self.A_S_Matrix2_masked=ma.masked_array(A_S_matrix2,A_S_Mask2)
         self.Px2_masked=ma.masked_array(self.Px2,self.U2_masked)
         self.Pz2_masked=ma.masked_array(self.Pz2,self.U2_masked)
         
         temp1,temp2,temp3=Matrix_TMOKE(self.chi_masked3,nx,gamma)
         A_S_matrix3,F_matrix3,self.Rx=temp1,temp2,temp3
         self.F_Matrix3_masked=ma.masked_array(F_matrix3,A_S_Mask3)
         self.A_S_Matrix3_masked=ma.masked_array(A_S_matrix3,A_S_Mask3)
         self.Rx_masked=ma.masked_array(self.Rx,self.U3_masked)
              
    def XM_define(self): 
        #this method has the purpose of moving through from the medium boundary matrices to the single interface reflection coefficients, using the formalism of
        #the Stepanov-Sinha paper. 
         a=self.A_S_Matrix1_masked
         a.data[a.mask==True]=0
         b=self.A_S_Matrix2_masked
         b.data[b.mask==True]=0
         c=self.A_S_Matrix3_masked
         c.data[c.mask==True]=0
         A_S_Matrix=np.ma.array(a.data+b.data+c.data,mask=list(map(and_,map(and_,a.mask,b.mask),c.mask))).data
         a=self.F_Matrix1_masked
         a.data[a.mask==True]=0
         b=self.F_Matrix2_masked
         b.data[b.mask==True]=0
         c=self.F_Matrix3_masked
         c.data[c.mask==True]=0
         F_Matrix=np.ma.array(a.data+b.data+c.data,mask=list(map(and_,map(and_,a.mask,b.mask),c.mask))).data
         theta=self.theta
         
         M=self.M
         
         AS1=A_S_Matrix[...,0:A_S_Matrix.shape[-1]-1]#upper
         AS2=A_S_Matrix[...,1:A_S_Matrix.shape[-1]]#lower
         F2=F_Matrix[...,1:A_S_Matrix.shape[-1]]
         #reflection coefficients are defined by multiplying two medium boundary matrices across a boundary
        
         M_tot=(M**2).sum(axis=0)
         
         if len(A_S_Matrix.shape)==3:
             permutation=(2,0,1)
         if len(A_S_Matrix.shape)==4:
             permutation=(2,3,0,1)
         if len(A_S_Matrix.shape)==5:
             permutation=(2,3,4,0,1)
         AS1=np.transpose(AS1, permutation)
         AS2=np.transpose(AS2, permutation)
         F2=np.transpose(F2, permutation)
         self.F2_mas=F2[...,0:2,0:2]
         self.F2_menos=F2[...,2:4,2:4]
         X=np.matmul(np.linalg.inv(AS1),AS2)
         #Equation taken from the line of text below Equation 58 in Stepanov Sinha
         Xtt=X[...,0:2,0:2]
         Xtr=X[...,0:2,2:4]
         Xrt=X[...,2:4,0:2]
         Xrr=X[...,2:4,2:4]
         self.Mtt=np.linalg.inv(self.F2_mas)@np.linalg.inv(Xtt)
         self.Mtr=-self.Mtt@Xtr@self.F2_menos
         self.Mrt=Xrt@np.linalg.inv(Xtt)
         self.Mrr=(Xrr-self.Mrt@Xtr)@self.F2_menos
         #Equation 60 of Stepanov Sinha
         #@jit
         
         #@njit
         def Mrt_mag(Xtt,Xrt,M_tot):
             #exports the Mrt matrix in a 
             Px1=self.Px1_masked
             Rx=self.Rx_masked
             Pz1=self.Pz1_masked
             u=self.U
             Basischange_Mask1=self.mask_expand(self.mask1,[2,2],before_after="after")
             Basischange_Mask2=self.mask_expand(self.mask2,[2,2],before_after="after")
             Basischange_Mask3=self.mask_expand(self.mask3,[2,2],before_after="after")
             
             Basischange1a=np.zeros(Basischange_Mask1.shape,dtype=complex)
             Basischange2a=np.zeros(Basischange_Mask1.shape,dtype=complex)
             Basischange3a=np.zeros(Basischange_Mask1.shape,dtype=complex)
             Basischange1b=np.zeros(Basischange_Mask1.shape,dtype=complex)
             Basischange2b=np.zeros(Basischange_Mask1.shape,dtype=complex)
             Basischange3b=np.zeros(Basischange_Mask1.shape,dtype=complex)
             ones=np.ones(M_tot.shape,dtype=complex)
             zeros=ones-ones
             Basischange1a[...,0,0]=ones
             Basischange1a[...,0,1]=ones
             Basischange1a[...,1,0]=Px1.data[0,...]*u[0,...]-Pz1.data[0,...]*np.cos(theta)
             Basischange1a[...,1,1]=Px1.data[1,...]*u[1,...]-Pz1.data[1,...]*np.cos(theta)
             #to change from the basis of the incoming light to the eigenbasis of the layer
             Basischange1b[...,0,0]=ones
             Basischange1b[...,0,1]=ones
             Basischange1b[...,1,0]=Px1.data[2,...]*u[2,...]-Pz1.data[2,...]*np.cos(theta)
             Basischange1b[...,1,1]=Px1.data[3,...]*u[3,...]-Pz1.data[3,...]*np.cos(theta)
             #to change from the eigenbasis of the layer to the eigenbasis of the outgoing light
             #these matrices are defined in equation 8 of the new paper
             Basischange1a=ma.masked_array(Basischange1a,Basischange_Mask1)
             Basischange1b=ma.masked_array(Basischange1b,Basischange_Mask1)
                       
                 
             Basischange2a[:,:,:,0,0]=ones
             Basischange2a[:,:,:,0,1]=zeros
             Basischange2a[:,:,:,1,0]=zeros
             Basischange2a[:,:,:,1,1]=ones
             Basischange2b[:,:,:,0,0]=ones
             Basischange2b[:,:,:,0,1]=zeros
             Basischange2b[:,:,:,1,0]=zeros
             Basischange2b[:,:,:,1,1]=ones
             
             Basischange2a=ma.masked_array(Basischange2a,Basischange_Mask2)
             Basischange2b=ma.masked_array(Basischange2b,Basischange_Mask2)
             
             Basischange3a[...,0,0]=ones
             Basischange3a[...,0,1]=zeros
             Basischange3a[...,1,0]=zeros
             Basischange3a[...,1,1]=Rx.data[1,...]*u[1,...]*np.cos(theta)+(np.cos(theta))**2
             #to change from the basis of the incoming light to the eigenbasis of the layer
             Basischange3b[...,0,0]=ones
             Basischange3b[...,0,1]=zeros
             Basischange3b[...,1,0]=zeros
             Basischange3b[...,1,1]=Rx.data[3,...]*u[3,...]*np.cos(theta)+(np.cos(theta))**2
             
             Basischange3a=ma.masked_array(Basischange3a,Basischange_Mask3)
             Basischange3b=ma.masked_array(Basischange3b,Basischange_Mask3)
             
             a=Basischange1a
             a.data[a.mask==True]=0
             b=Basischange2a
             b.data[b.mask==True]=0
             c=Basischange3a
             c.data[c.mask==True]=0
             Basischange1=np.ma.array(a.data+b.data+c.data,mask=list(map(and_,map(and_,a.mask,b.mask),c.mask))).data
             
             a=Basischange1b
             a.data[a.mask==True]=0
             b=Basischange2b
             b.data[b.mask==True]=0
             c=Basischange3b
             c.data[c.mask==True]=0
             Basischange2=np.ma.array(a.data+b.data+c.data,mask=list(map(and_,map(and_,a.mask,b.mask),c.mask))).data
             
             
             
             #for the non-magnetic layers, the basischange matrix is simply the identity matrix
             #@jit
             def Matrixalgebra(Basischange,Basischange2,Xrt,Xtt):
                 Eigen_reflection=np.matmul(Xrt,np.linalg.inv(Xtt))         
                 processed_matrices=np.matmul(np.matmul(Basischange2[:,:,0:-1,:,:],Eigen_reflection),np.linalg.inv(Basischange[:,:,0:-1,:,:]))
                 return processed_matrices
             Mrt_3D=Matrixalgebra(Basischange1,Basischange2,Xrt,Xtt)
             #simply applying equation 62 of Stepanov Sinha, and applying the basischange
             return Mrt_3D,Basischange1,Basischange2
         
         
         self.Mrt_matrix,self.Basischange1,self.Basischange2=Mrt_mag(Xtt,Xrt,M_tot)
         
         return self.Mrt_matrix
     #this is the main routine called by the XRMS simulation loop.
    def Matrix_stack(self,stack_coordinates=[0,0]):
        
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
        tempzeros=np.array([[0.],[0.]])
        M_list_tt.append(self.Mtt[stack_coordinates[0],stack_coordinates[1],0,:,:])
        M_list_tr.append(self.Mtr[stack_coordinates[0],stack_coordinates[1],0,:,:]*roughness_reduction)
        M_list_rt.append(self.Mrt[stack_coordinates[0],stack_coordinates[1],0,:,:]*roughness_reduction)#Aplicando la rugosidad solamente a las partes relacionados a la reflexión
        M_list_rr.append(self.Mrr[stack_coordinates[0],stack_coordinates[1],0,:,:])
        W_list_tt.append(self.Mtt[stack_coordinates[0],stack_coordinates[1],0,:,:])
        W_list_tr.append(self.Mtr[stack_coordinates[0],stack_coordinates[1],0,:,:]*roughness_reduction)
        W_list_rt.append(self.Mrt[stack_coordinates[0],stack_coordinates[1],0,:,:]*roughness_reduction)
        W_list_rr.append(self.Mrr[stack_coordinates[0],stack_coordinates[1],0,:,:])
        R_fields.append(tempzeros)
        T_fields.append(tempzeros)
        for i in range(1,len(self.z)-1):
            #print(params[i])
            
            #Selección de parametros elementales
            Mtt,Mtr,Mrt,Mrr=self.Mtt[stack_coordinates[0],stack_coordinates[1],i,:,:],self.Mtr[stack_coordinates[0],stack_coordinates[1],i,:,:],self.Mrt[stack_coordinates[0],stack_coordinates[1],i,:,:],self.Mrr[stack_coordinates[0],stack_coordinates[1],i,:,:]
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
        for i in range(len(self.z)-3,-1,-1):
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
        
        Basischange1=self.Basischange1[stack_coordinates[0],stack_coordinates[1],1:,:,:]
        Basischange2=self.Basischange2[stack_coordinates[0],stack_coordinates[1],1:,:,:]
        T_fields_linear=np.zeros(self.T_fields.shape,dtype=complex)
        R_fields_linear=np.zeros(self.R_fields.shape,dtype=complex)
        for l in range(Basischange1.shape[0]):
            T_fields_linear[l,:,:]=(Basischange1[l,:,:])@self.T_fields[l,:]
            R_fields_linear[l,:,:]=(Basischange2[l,:,:])@self.R_fields[l,:]    
        self.T_fields_linear=np.array(T_fields_linear)
        self.R_fields_linear=np.array(R_fields_linear)
        
        temp=self.Mrt-self.Mrt
        
        self.efields=np.zeros((temp.shape[2],4,3),dtype=complex)
        Px1=self.Px1_masked
        Rx=self.Rx_masked
        Pz1=self.Pz1_masked
        for j in range (4):
            for l in range(temp.shape[2]):
                if self.mask1[0,0,l]==False:
                    self.efields[l,j,0]=Px1[j,0,0,l]
                    self.efields[l,j,1]=1.
                    self.efields[l,j,2]=Pz1[j,0,0,l]
        for j in range (4):
            for l in range(temp.shape[2]):
                if self.mask2[0,0,l]==False:
                    if j%2==0:
                        self.efields[l,j,0]=0
                        self.efields[l,j,1]=1
                        self.efields[l,j,2]=0
                    else:
                        self.efields[l,j,0]=np.sin(self.theta)
                        self.efields[l,j,1]=0
                        self.efields[l,j,2]=np.cos(self.theta)
        for j in range (4):
            for l in range(temp.shape[2]):
                if self.mask3[0,0,l]==False:
                    if j%2==0:
                        self.efields[l,j,0]=0
                        self.efields[l,j,1]=1
                        self.efields[l,j,2]=0
                    else:
                        self.efields[l,j,0]=Rx[j,0,0,l]*np.cos(self.theta)
                        self.efields[l,j,1]=0
                        self.efields[l,j,2]=np.cos(self.theta)
        
        #outputting the fields in the sigma-pi basis 
    def get_R(self):
        sample=self.sample
        na=sample.na
        
        r0=self.r0
        att2=self.T_fields_linear
        lamda=self.lamda
        att3=att2*np.conj(att2)
        
        att_temp=att3.sum(axis=1)/2
        
        if sample.sim_type=="Crystal":
            multiplier=lamda**2*r0/np.pi
            f_bulk=sample.f_bulk
            chi_zero=na[0,0,:]*multiplier*f_bulk
            u_bulk=(chi_zero+np.sin(self.theta)**2)**0.5
            att2=np.exp(-complex(0,1)*u_bulk*2*np.pi/self.lamda*self.z)
            att_temp=att2*np.conj(att2)
        #The reflection coefficients are to be scaled by the square of the relative field strength within the sample. We
        #scale by the square to include attenuation of both incident and reflected beams. In the crystal case, we approximate
        #the attenuation via a scalar model to avoid artificial Bragg enhancement from the micromagnetic simulation lattice
        
        if sample.sim_type=="Multilayer":
            att=np.ones(len(att_temp)+1,dtype=complex)
            att[1:]=att_temp.squeeze()
        if self.sample.sim_type=="Crystal":
            att=att_temp[1:]
            self.att_factor=att[2*self.sample.n_unique]/att[self.sample.n_unique]
        R_array=np.zeros(self.Mrt_matrix.shape,dtype=complex)
        for l in range(att.shape[0]-1):
            R_array[:,:,l,:,:]=self.Mrt_matrix[:,:,l,:,:]*att[l]
        self.R_array=R_array
        
    def get_Faraday_rotated_stripe(self):
        #This method takes a set of reflection coefficients along a line, and calculates
        #the corresponding diffraction pattern taking into account differential absorption and Ewald 
        #sphere curvature
        
        #please note that the magnetization values must also be rotated due to the vectorial nature of the physical process being 
        #modelled. 
        M_long_in, M_long_out, M_pi_in,M_pi_out,M_trans=self.get_M_Beamdirection()
        R_shape=self.R_array.shape[0:2]
        R_short=np.argmin(R_shape)
        self.phi_rotate=self.sample.phi_rotate*np.pi/180
        if R_short==0:
            R_stripe=self.R_array[0,:,:,:,:]
            M_long_in=M_long_in[0,:,:]
            M_long_out=M_long_out[0,:,:]
            M_pi_in=M_pi_in[0,:,:]
            M_pi_out=M_pi_out[0,:,:]
            M_trans=M_trans[0,:,:]
        if R_short==1:
            R_stripe=self.R_array[:,0,:,:,:]
            M_long_in=M_long_in[:,0,:]
            M_long_out=M_long_out[:,0,:]
            M_pi_in=M_pi_in[:,0,:]
            M_pi_out=M_pi_out[:,0,:]
            M_trans=M_trans[:,0,:]
        
        theta=self.theta
        phi=self.phi_rotate
        
        def get_abs_adj(XMCD_XMLD,M,in_out):
            #to calculate the adjusted absorption
            #the interpolation is assumed to be along the x direction
            #which in the zero rotation case is assumed to be constant in the x
            #direction
            ny=self.size_y
            nz=len(self.sample.z)
            
            dx=self.sample.dx
            dz=self.sample.dz
            
            dz_mag=self.sample.dz_mag
            if self.sample.sim_type=="Crystal":
                ratio=dx/(dz/self.sample.n_unique)
            else:
                ratio=dx/dz
            y=np.linspace(0,ny-1,ny)
            z=np.linspace(0,nz-1,nz)
            #we are talking here about the pixel in z
            interp_plus=np.zeros((len(y),len(z)),dtype=complex)
            interp_minus=np.zeros((len(y),len(z)),dtype=complex)
            for k in range(len(z)):
                trans_plus=(y+z[k]/ratio/np.tan(theta)*np.sin(phi))%(ny-1)
                trans_minus=(y-z[k]/ratio/np.tan(theta)*np.sin(phi))%(ny-1)
                #we can caluculate the projected magnetization through the sample by shifting each plane in the multilayer sample along the longitudinal
                #direction
                pts_plus=[]
                pts_minus=[]
                for i in range (len(y)):
                    point_plus=trans_plus[i]
                    point_minus=trans_minus[i]
                    pts_plus.append(point_plus)
                    pts_minus.append(point_minus)
                pts_plus=np.array(pts_plus)
                pts_minus=np.array(pts_minus)
                average_mx=np.sum(M[:,k])/(M.shape[0])
                average_mx2=np.sum(M[:,k]**2)/(M.shape[0])
                if XMCD_XMLD=="XMCD":
                    a=np.interp(pts_plus,y, M[:,k], left=None, right=None, period=ny)
                    b=np.interp(pts_minus,y, M[:,k], left=None, right=None, period=ny)
                if XMCD_XMLD=="XMLD":
                    a=np.interp(pts_plus,y, M[:,k]**2, left=None, right=None, period=ny)
                    b=np.interp(pts_minus,y, M[:,k]**2, left=None, right=None, period=ny)
                a=np.interp(pts_plus,y, M[:,k], left=None, right=None, period=ny)
                b=np.interp(pts_minus,y, M[:,k], left=None, right=None, period=ny)
                interp_plus[:,k]=a
                interp_minus[:,k]=b
                
            partial_sum_plus=interp_plus-interp_plus
            partial_sum_minus=interp_plus-interp_plus
            
            
            for j in range(interp_plus.shape[1]):
                if j==0:
                    partial_sum_plus[:,j]=interp_plus[:,j]
                    partial_sum_minus[:,j]=interp_minus[:,j]
                else:
                    partial_sum_plus[:,j]=partial_sum_plus[:,j-1]+interp_plus[:,j]
                    partial_sum_minus[:,j]=partial_sum_minus[:,j-1]+interp_minus[:,j]
                #the plus and minus are for the incoming and outgoing beams respectively. We need partial sums as well
                #to include the total contributions up to each layer where reflection is to be calculated
            for k in range(len(z)):
                    
                a=np.interp(pts_plus,y, partial_sum_minus[:,k], left=None, right=None, period=ny)
                b=np.interp(pts_minus,y, partial_sum_plus[:,k], left=None, right=None, period=ny)
                partial_sum_plus[:,k]=b
                partial_sum_minus[:,k]=a
            #here we unwrap the partial sums so that they correspond to the correct spatial position of the reflections
            
            u,umag,umag2=self.get_U_mag_absorption()   
            x=np.exp(-np.imag(u)*2*np.pi/self.lamda*dz_mag)
            dx=np.exp(-np.imag(umag)*2*np.pi/self.lamda*dz_mag)-x
            dx2=np.exp(-np.imag(umag2)*2*np.pi/self.lamda*dz_mag)-x
            x=x
            dx=dx
            dx2=dx2
            #calculating relative differential absorptions to first order
            
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
        self.adj_abs_plus_XMCD=get_abs_adj(XMCD_XMLD="XMCD",M=M_long_in,in_out="in")
        self.adj_abs_minus_XMCD=get_abs_adj(XMCD_XMLD="XMCD",M=M_long_out,in_out="out")
        self.adj_abs_plus_XMLD_pi=get_abs_adj(XMCD_XMLD="XMLD",M=M_pi_in,in_out="in")
        self.adj_abs_minus_XMLD_pi=get_abs_adj(XMCD_XMLD="XMLD",M=M_pi_out,in_out="out")
        self.adj_abs_plus_XMLD_trans=get_abs_adj(XMCD_XMLD="XMLD",M=M_trans,in_out="in")
        self.adj_abs_minus_XMLD_trans=get_abs_adj(XMCD_XMLD="XMLD",M=M_trans,in_out="out")
        
        
        
    def get_M_Beamdirection(self):
        #go from M defined in x, y and z to an M defined as longitudinal (along)
        #the beam, sigma and pi
        
        #This routine is needed for the inclusion of the differential absorption due to magnetization along the x direction (XMCD) 
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
        f_charge=self.f_charge_maglayer_scalar
        f_Mag=self.f_Mag_scalar
        f_Mag2=np.max(self.f_Mag2)
        r0=self.r0
        na=self.na_scalar
               
        multiplier=lamda**2*r0/np.pi
        chi_zero=na*multiplier*f_charge
        u=(chi_zero+np.sin(self.theta)**2)**0.5
        chi_zero=na*multiplier*(f_charge+f_Mag)
        umag=(chi_zero+np.sin(self.theta)**2)**0.5
        chi_zero=na*multiplier*(f_charge+f_Mag2)  
        umag2=(chi_zero+np.sin(self.theta)**2)**0.5 
        
        return u,umag,umag2
      
        
    def get_Faraday_Parallel(self):
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
            #to calculate the adjusted absorption
            nx=self.size_x
            ny=self.size_y
            nz=len(self.sample.z)
            
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
                x_shifted_plus=(x+z[k]/ratio/np.tan(theta))%(nx-1)
                x_shifted_minus=(x-z[k]/ratio/np.tan(theta))%(nx-1)
                #we can caluculate the projected magnetization through the sample by shifting each plane in the multilayer sample along the longitudinal
                #direction
                pts_plus=[]
                pts_minus=[]
                for i in range (len(x)):
                    for j in range (len(y)):
                        point_plus=[x_shifted_plus[i],y[j]]
                        point_minus=[x_shifted_minus[i],y[j]]
                        pts_plus.append(point_plus)
                        pts_minus.append(point_minus)
                average_mx=np.sum(M[:,:,k])/(M.shape[0]*M.shape[1])
                average_mx2=np.sum(M[:,:,k]**2)/(M.shape[0]*M.shape[1])
                if XMCD_XMLD=="XMCD":
                    temp_plus = RegularGridInterpolator((x, y), M[:,:,k],bounds_error=False, fill_value=average_mx)
                    temp_minus = RegularGridInterpolator((x, y), M[:,:,k],bounds_error=False, fill_value=average_mx) 
                if XMCD_XMLD=="XMLD":
                    temp_plus = RegularGridInterpolator((x, y), M[:,:,k]**2,bounds_error=False, fill_value=average_mx2)
                    temp_minus = RegularGridInterpolator((x, y), M[:,:,k]**2,bounds_error=False, fill_value=average_mx2) 
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
                #the plus and minus are for the incoming and outgoing beams respectively. We need partial sums as well
                #to include the total contributions up to each layer where reflection is to be calculated
            for k in range(len(z)):
                    
                temp_plus = RegularGridInterpolator((x, y), partial_sum_minus[:,:,k],bounds_error=False, fill_value=np.sum(partial_sum_minus[0,:,k])/M.shape[1])
                temp_minus = RegularGridInterpolator((x, y), partial_sum_plus[:,:,k],bounds_error=False, fill_value=np.sum(partial_sum_minus[0,:,k])/M.shape[1]) 
                a=np.reshape(np.array(temp_plus(pts_plus)),(len(x),len(y)))
                b=np.reshape(np.array(temp_minus(pts_minus)),(len(x),len(y)))
                partial_sum_plus[:,:,k]=b
                partial_sum_minus[:,:,k]=a
            #here we unwrap the partial sums so that they correspond to the correct spatial position of the reflections
            
            u,umag,umag2=self.get_U_mag_absorption()   
            x=np.exp(-np.imag(u)*2*np.pi/self.lamda*dz_mag)
            dx=np.exp(-np.imag(umag)*2*np.pi/self.lamda*dz_mag)-x
            dx2=np.exp(-np.imag(umag2)*2*np.pi/self.lamda*dz_mag)-x
            
            #calculating relative differential absorptions to first order
            
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
        self.adj_abs_plus_XMCD=get_abs_adj(XMCD_XMLD="XMCD",M=M_long_in,in_out="in")
        self.adj_abs_minus_XMCD=get_abs_adj(XMCD_XMLD="XMCD",M=M_long_out,in_out="out")
        self.adj_abs_plus_XMLD_pi=get_abs_adj(XMCD_XMLD="XMLD",M=M_pi_in,in_out="in")
        self.adj_abs_minus_XMLD_pi=get_abs_adj(XMCD_XMLD="XMLD",M=M_pi_out,in_out="out")
        self.adj_abs_plus_XMLD_trans=get_abs_adj(XMCD_XMLD="XMLD",M=M_trans,in_out="in")
        self.adj_abs_minus_XMLD_trans=get_abs_adj(XMCD_XMLD="XMLD",M=M_trans,in_out="out")
        #absorption factor        
        
        
    def Ewald_sphere_pixel_index(self,input_array,sample, simulation_input):
        #From the array of reflection coefficients, get the Ewald sphere slice
            if "total_thickness" not in simulation_input['3D_Sample_Parameters'].keys():          
                thickness=sample.z[-1]
            else:
                thickness=simulation_input['3D_Sample_Parameters']["total_thickness"]
            dim_x=sample.dx*input_array.shape[0]*1e-9
            dim_y=sample.dy*input_array.shape[1]*1e-9
            nx=input_array.shape[0]
            ny=input_array.shape[1]
            
            
            lamda=self.lamda
            pixel_thetax=np.arcsin(lamda/dim_x)#FFT output angle of each diffraction order (Fraunhofer diffraction theory)
            pixel_thetay=np.arcsin(lamda/dim_y)
            pixel_thetaz=np.arcsin(lamda/(2*thickness))#the factor of two is for the reflection
            #theta values corresponding to the natural fft output
            
            #pixel numbers of the input array
            if sample.sim_type=="Crystal":
                UC_per_mm_cell=np.int64(sample.dz*1e-9/sample.unit_cell)
                temp=(simulation_input['3D_Sample_Parameters']["shape"][2]-1)/(simulation_input['3D_Sample_Parameters']["shape"][2])*(1/(2*UC_per_mm_cell*(simulation_input['3D_Sample_Parameters']["shape"][2]))+1)    
                points_z_fft=self.z[1:]/(self.z[-1]-self.z[1])*2*np.pi*temp-np.pi
            if sample.sim_type=="Multilayer":
                points_z_fft=self.z[1:]/(self.z[-1]-self.z[1])*2*np.pi-np.pi
            points_x_fft=np.linspace(-np.pi,np.pi-2*np.pi/nx,nx)
            points_y_fft=np.linspace(-np.pi,np.pi-2*np.pi/ny,ny)
            
            xv, yv, zv = np.meshgrid(points_x_fft, points_y_fft, points_z_fft, indexing='ij')
            #scaling the z positions for the non uniform fft. The effective array size in z is set as the 
            #thickness of the sample
            delta_qz=2*np.pi/thickness
            
            #q spacing of a pixel in the FT output
            qz_origin=2*np.pi/lamda*np.sin(self.theta)
            #centre of the Ewald sphere in terms of qz        
            qz_scatter_pixel=2*qz_origin/delta_qz
            #pixel value in z of the centre of the scattered light            
            det_pixel=simulation_input['Simulation_Parameters']['det_dx']
            detector_distance=simulation_input['Simulation_Parameters']["det_sample_distance"]
            n_det_x=simulation_input['Simulation_Parameters']['det_size'][0]
            n_det_y=simulation_input['Simulation_Parameters']['det_size'][1]
            order_x=simulation_input['Simulation_Parameters']['orders_x']
            order_y=simulation_input['Simulation_Parameters']['orders_y']
            
            detector_pixel_angle=np.arctan(det_pixel/detector_distance)
            #the physical angle represented by each detector pixel
            detpixels_per_fourier_pixel_y=pixel_thetay/detector_pixel_angle
            detpixels_per_fourier_pixel_x=pixel_thetax/detector_pixel_angle
            #the number of detector pixels per diffraction order. This does not yet include the 
            #factor of 1/cos(theta)
            
            if n_det_y%2==0:
                detector_y=np.linspace(-n_det_y/2,n_det_y/2-1,n_det_y)*det_pixel
            else:
                detector_y=np.linspace(-(n_det_y-1)/2,(n_det_y-1)/2,n_det_y)*det_pixel
            if n_det_x%2==0:
                detector_xz=np.linspace(-n_det_x/2,n_det_x/2-1,n_det_x)*det_pixel
            else:
                detector_xz=np.linspace(-(n_det_x-1)/2,(n_det_x-1)/2,n_det_x)*det_pixel
            
            detector_angle_y=np.arctan(detector_y/detector_distance)#angle of each detector pixel
            detector_angle_xz=np.arctan(detector_xz/detector_distance)
            #detector angles for each detector pixel            
            
            detector_pixel_y=detector_angle_y/pixel_thetay
            detector_pixel_x=detector_angle_xz/pixel_thetax*np.sin(self.theta)         
            detector_pixel_z=detector_angle_xz/pixel_thetaz*np.cos(self.theta)+qz_scatter_pixel
            
            ones_y=np.ones(n_det_y)
            ones_xz=np.ones(n_det_x)
            det_z=np.transpose(np.outer(ones_y,detector_pixel_z))
            
            detector_pixel_y=np.linspace(-order_y,order_y,2*order_y+1)
            detector_pixel_x=np.linspace(-order_x,order_x,2*order_x+1)
            
            delta_qx_physical=4*np.pi/dim_x
            qz_physical=2*np.pi/lamda*np.sin(np.arccos((2*np.pi/lamda*np.cos(self.theta)+np.linspace(-order_x,order_x,2*order_x+1)*delta_qx_physical)/(2*np.pi/lamda)))+qz_origin
            qz_pixel=qz_physical/delta_qz
            qz_pixel[np.isnan(qz_physical)==True]=qz_pixel[np.int64(0.5*(len(qz_pixel)-1))]
            qx_physical=(qz_physical-qz_origin*2)*np.tan(self.theta)
            
            detector_pixel_x[np.isnan(qz_physical)==True]=0
            ones_y=np.ones(2*order_y+1)
            det_z=np.transpose(np.outer(ones_y,qz_pixel))
            
            qy_pixel=detector_pixel_y
            order_pixels_y=np.round((qy_pixel-qy_pixel[len(qy_pixel)//2])*detpixels_per_fourier_pixel_y+(n_det_y)/2)
            order_pixels_xz=np.round(-qx_physical/delta_qx_physical/np.sin(self.theta)*detpixels_per_fourier_pixel_x+n_det_x/2)
            
            array_x, array_y,array_z=np.meshgrid(detector_pixel_x,  detector_pixel_y,self.z/thickness,indexing='ij')
            qz_pixel_array=np.zeros((2*order_x+1,2*order_y+1))
            for j in range(2*order_x+1):
                qz_pixel_array[j,:]=np.ones(2*order_y+1)*qz_pixel[j]
                
            R_FFT2D=np.fft.fftshift(np.fft.fftn(input_array,axes=[0,1]),axes=[0,1])
            ROI=R_FFT2D[nx//2-order_x:nx//2+order_x+1,ny//2-order_y:ny//2+order_y+1,:,:,:]
            R_FFT2D_diff1=R_FFT2D-R_FFT2D
            R_FFT2D_diff2=R_FFT2D-R_FFT2D
            
            if simulation_input['Simulation_Parameters']["differential_absorption"]==True:
                permutation=(3,4,0,1,2)
                temp1=np.transpose(input_array,permutation)
                R_FFT2D_diff1=np.fft.fftshift(np.fft.fftn(temp1*(self.adj_abs_plus_XMCD[:,:,1:]*self.adj_abs_minus_XMCD[:,:,1:]),axes=[2,3]),axes=[2,3])
                R_FFT2D_diff2=np.fft.fftshift(np.fft.fftn(temp1/(self.adj_abs_plus_XMCD[:,:,1:]*self.adj_abs_minus_XMCD[:,:,1:]),axes=[2,3]),axes=[2,3])
                #strong absorption, weak reflection 
                permutation=(2,3,4,0,1)
                R_FFT2D_diff1=np.transpose(R_FFT2D_diff1,permutation)
                R_FFT2D_diff2=np.transpose(R_FFT2D_diff2,permutation)
                ROI_diff1=R_FFT2D_diff1[nx//2-order_x:nx//2+order_x+1,ny//2-order_y:ny//2+order_y+1,:,:,:]
                ROI_diff2=R_FFT2D_diff2[nx//2-order_x:nx//2+order_x+1,ny//2-order_y:ny//2+order_y+1,:,:,:]
                
            Phase_Fourier=np.zeros(array_z.shape,dtype=complex)
            
            for j in range(len(self.z)):
                Phase_Fourier[:,:,j]=np.exp(-2*np.pi*complex(0,1)*array_z[:,:,j]*qz_pixel_array)
            #calculating the phase factors at the Ewald sphere for the irregular Fourier transform
            
            FT_Ewald=np.zeros((2*order_x+1,2*order_y+1,2,2),dtype=complex)
            if simulation_input['Simulation_Parameters']["differential_absorption"]==True:
                FT_Ewald1=np.zeros((2*order_x+1,2*order_y+1,2,2),dtype=complex)
                FT_Ewald2=np.zeros((2*order_x+1,2*order_y+1,2,2),dtype=complex)
            
            for j in range(2):
                for k in range(2):                        
                    FT_Ewald[:,:,j,k]=np.sum(Phase_Fourier[:,:,0:-1]*ROI[:,:,:,j,k],axis=2)
                    if simulation_input['Simulation_Parameters']["differential_absorption"]==True:
                        FT_Ewald1[:,:,j,k]=np.sum(Phase_Fourier[:,:,0:-1]*ROI_diff1[:,:,:,j,k],axis=2)
                        FT_Ewald2[:,:,j,k]=np.sum(Phase_Fourier[:,:,0:-1]*ROI_diff2[:,:,:,j,k],axis=2)    
                         
            output_full1=np.zeros((n_det_x,n_det_y,2,2),dtype=complex)
            output_full2=np.zeros((n_det_x,n_det_y,2,2),dtype=complex)            
            
                       
            
            if sample.sim_type=="Crystal":
                
                kernel=np.ones(UC_per_mm_cell,dtype=complex)
                for j in range(UC_per_mm_cell):
                    kernel[j]=self.att_factor**(j/UC_per_mm_cell)
                #the kernel is used to convolve the output from the FT over the micromagnetic cell with an object with unit cell periodicity, in order
                #to generate the equivalent FT from the full object
                 
                Phase_kernel=np.zeros((2*order_x+1,2*order_y+1,len(kernel)),dtype=complex)
                for j in range(len(kernel)):
                    Phase_kernel[:,:,j]=np.exp(-2*np.pi*complex(0,1)*j*sample.unit_cell/thickness*qz_pixel_array)
                f_kernel =np.sum(Phase_kernel*kernel,axis=2)
                           
                #applies periodic boundary conditions, expanding a Fourier output to 3X the area
            if simulation_input['Simulation_Parameters']["differential_absorption"]==False:
                self.output=np.zeros((det_z.shape[0],det_z.shape[1],2,2),dtype=complex)
                for j in range(2):
                    for k in range(2):
                        f = FT_Ewald[:,:,j,k]
                        
                        if sample.sim_type!="Crystal":
                            self.output[:,:,j,k]= np.reshape(f,(self.output.shape[0],self.output.shape[1])) 
                        if sample.sim_type=="Crystal":
                            self.output[:,:,j,k]= np.reshape(f*f_kernel,(self.output.shape[0],self.output.shape[1]))
                        
                
                for j in range(order_x*2+1):
                    for k in range(order_y*2+1):
                        if (order_pixels_xz[j] in range(n_det_x)) and (order_pixels_y[k] in range(n_det_y)):
                            output_full1[int(order_pixels_xz[j]),int(order_pixels_y[k]),:,:]=self.output[j,k,:,:]
# Placing the output into the corresponding detector pixels depending upon the experimental geometry                            
                self.output=output_full1
                    
                return self.output,self.output  
            
            if simulation_input['Simulation_Parameters']["differential_absorption"]==True:
                self.output1=np.zeros((det_z.shape[0],det_z.shape[1],2,2),dtype=complex)
                self.output2=np.zeros((det_z.shape[0],det_z.shape[1],2,2),dtype=complex)
                
                for j in range(2):
                    for k in range(2):
                        f1 = FT_Ewald1[:,:,j,k]
                        f2 = FT_Ewald2[:,:,j,k]
                        
                        if sample.sim_type!="Crystal":
                            self.output1[:,:,j,k]= np.reshape(f1,(self.output1.shape[0],self.output1.shape[1]))
                            self.output2[:,:,j,k]= np.reshape(f2,(self.output1.shape[0],self.output1.shape[1]))
                        if sample.sim_type=="Crystal":
                            self.output1[:,:,j,k]= np.reshape(f1*f_kernel,(self.output1.shape[0],self.output1.shape[1]))
                            self.output2[:,:,j,k]= np.reshape(f2*f_kernel,(self.output2.shape[0],self.output2.shape[1]))
                
                for j in range(order_x*2+1):
                    for k in range(order_y*2+1):
                        if (order_pixels_xz[j] in range(n_det_x)) and (order_pixels_y[k] in range(n_det_y)):
                            output_full1[int(order_pixels_xz[j]),int(order_pixels_y[k]),:,:]=self.output1[j,k,:,:]
                            output_full2[int(order_pixels_xz[j]),int(order_pixels_y[k]),:,:]=self.output2[j,k,:,:]
                self.output1=output_full1   
                self.output2=output_full2
                    
                return self.output1,self.output2 

    def Ewald_sphere_rotated_stripe(self,input_stripe,sample, simulation_input):
            if "output_2D" in simulation_input['Simulation_Parameters'].keys():    
                output_2D=simulation_input['Simulation_Parameters']["output_2D"]
            else:
                output_2D=False
        #From the array of reflection coefficients, get the Ewald sphere slice
            if simulation_input['Simulation_Parameters']["differential_absorption"]==False:
                self.phi_rotate=self.sample.phi_rotate*np.pi/180
            if "total_thickness" not in simulation_input['3D_Sample_Parameters'].keys():          
                thickness=sample.z[-1]
            else:
                thickness=simulation_input['3D_Sample_Parameters']["total_thickness"]
            dim_y=sample.dy*input_stripe.shape[0]*1e-9
            dim_x=dim_y
            ny=input_stripe.shape[0]
            
            
            lamda=self.lamda
            pixel_thetay=np.arcsin(lamda/dim_y)
            pixel_thetaz=np.arcsin(lamda/(2*thickness))#the factor of two is for the reflection
            #theta values corresponding to the natural fft output
            
            #pixel numbers of the input array
            if sample.sim_type=="Crystal":
                UC_per_mm_cell=np.int64(sample.dz*1e-9/sample.unit_cell)
                temp=(simulation_input['3D_Sample_Parameters']["shape"][2]-1)/(simulation_input['3D_Sample_Parameters']["shape"][2])*(1/(2*UC_per_mm_cell*(simulation_input['3D_Sample_Parameters']["shape"][2]))+1)    
                points_z_fft=self.z[1:]/(self.z[-1]-self.z[1])*2*np.pi*temp-np.pi
            if sample.sim_type=="Multilayer":
                points_z_fft=self.z[1:]/(self.z[-1]-self.z[1])*2*np.pi-np.pi
            points_y_fft=np.linspace(-np.pi,np.pi-2*np.pi/ny,ny)
            
            yv, zv = np.meshgrid(points_y_fft, points_z_fft, indexing='ij')
            #scaling the z positions for the non uniform fft. The effective array size in z is set as the 
            #thickness of the sample
            delta_qz=2*np.pi/thickness
            
            #q spacing of a pixel in the FT output
            qz_origin=2*np.pi/lamda*np.sin(self.theta)
            #centre of the Ewald sphere in terms of qz        
            qz_scatter_pixel=2*qz_origin/delta_qz
            #pixel value in z of the centre of the scattered light            
            det_pixel=simulation_input['Simulation_Parameters']['det_dx']
            detector_distance=simulation_input['Simulation_Parameters']["det_sample_distance"]
            n_det_x=simulation_input['Simulation_Parameters']['det_size'][0]
            n_det_y=simulation_input['Simulation_Parameters']['det_size'][1]
            order_y=simulation_input['Simulation_Parameters']['orders_y']
            order_x=order_y
            
            detector_pixel_angle=np.arctan(det_pixel/detector_distance)
            #the physical angle represented by each detector pixel
            detpixels_per_fourier_pixel_y=pixel_thetay/detector_pixel_angle
            #the number of detector pixels per diffraction order. This does not yet include the 
            #factor of 1/cos(theta)
            
            if n_det_y%2==0:
                detector_y=np.linspace(-n_det_y/2,n_det_y/2-1,n_det_y)*det_pixel
            else:
                detector_y=np.linspace(-(n_det_y-1)/2,(n_det_y-1)/2,n_det_y)*det_pixel
            if n_det_x%2==0:
                detector_xz=np.linspace(-n_det_x/2,n_det_x/2-1,n_det_x)*det_pixel
            else:
                detector_xz=np.linspace(-(n_det_x-1)/2,(n_det_x-1)/2,n_det_x)*det_pixel
            
            detector_angle_y=np.arctan(detector_y/detector_distance)#angle of each detector pixel
            detector_angle_xz=np.arctan(detector_xz/detector_distance)
            #detector angles for each detector pixel            
            
            detector_pixel_y=detector_angle_y/pixel_thetay
                     
            detector_pixel_z=detector_angle_xz/pixel_thetaz*np.cos(self.theta)+qz_scatter_pixel
           
            ones_y=np.ones(n_det_y)
            ones_xz=np.ones(n_det_x)
            det_z=np.transpose(np.outer(ones_y,detector_pixel_z))
            
            detector_pixel_y=np.linspace(-order_y,order_y,2*order_y+1)
                        
            delta_qx_physical=4*np.pi/dim_x
            qz_physical=2*np.pi/lamda*np.sin(np.arccos((2*np.pi/lamda*np.cos(self.theta)+np.sin(self.phi_rotate)*detector_pixel_y*delta_qx_physical)/(2*np.pi/lamda)))+qz_origin
            qz_pixel=qz_physical/delta_qz
            qz_pixel[np.isnan(qz_physical)==True]=qz_pixel[np.int64(0.5*(len(qz_pixel)-1))]
            qx_physical=(qz_physical-qz_origin*2)*np.tan(self.theta)
            
            
            detector_pixel_y[np.isnan(qz_physical)==True]=0
            
            det_z=qz_pixel
            
            qy_pixel=detector_pixel_y
            order_pixels_y=np.round((qy_pixel-qy_pixel[len(qy_pixel)//2])*detpixels_per_fourier_pixel_y*np.cos(self.phi_rotate)+(n_det_y)/2)
            order_pixels_xz=np.round(qx_physical/delta_qx_physical/np.sin(self.theta)*detpixels_per_fourier_pixel_y+n_det_x/2)
            #print(qx_physical/delta_qx_physical/np.sin(self.theta)*detpixels_per_fourier_pixel_y+n_det_x/2)
            array_y,array_z=np.meshgrid(detector_pixel_y,self.z/thickness,indexing='ij')
            qz_pixel_array=np.zeros((2*order_y+1))
            
            qz_pixel_array=qz_pixel
                
            R_FFT2D=np.fft.fftshift(np.fft.fftn(input_stripe,axes=[0]),axes=[0])
            ROI=R_FFT2D[ny//2-order_y:ny//2+order_y+1,:,:,:]
            R_FFT2D_diff1=R_FFT2D-R_FFT2D
            R_FFT2D_diff2=R_FFT2D-R_FFT2D
            
            if simulation_input['Simulation_Parameters']["differential_absorption"]==True:
                permutation=(2,3,0,1)
                temp1=np.transpose(input_stripe,permutation)
                R_FFT2D_diff1=np.fft.fftshift(np.fft.fftn(temp1*(self.adj_abs_plus_XMCD[:,1:]*self.adj_abs_minus_XMCD[:,1:]),axes=[2]),axes=[2])
                R_FFT2D_diff2=np.fft.fftshift(np.fft.fftn(temp1/(self.adj_abs_plus_XMCD[:,1:]*self.adj_abs_minus_XMCD[:,1:]),axes=[2]),axes=[2])
                permutation=(2,3,0,1)
                R_FFT2D_diff1=np.transpose(R_FFT2D_diff1,permutation)
                R_FFT2D_diff2=np.transpose(R_FFT2D_diff2,permutation)
                ROI_diff1=R_FFT2D_diff1[ny//2-order_y:ny//2+order_y+1,:,:,:]
                ROI_diff2=R_FFT2D_diff2[ny//2-order_y:ny//2+order_y+1,:,:,:]
                
            Phase_Fourier=np.zeros(array_z.shape,dtype=complex)
            
            for j in range(len(self.z)):
                Phase_Fourier[:,j]=np.exp(-2*np.pi*complex(0,1)*array_z[:,j]*qz_pixel_array)
            #calculating the phase factors at the Ewald sphere for the irregular Fourier transform
            
            FT_Ewald=np.zeros((2*order_y+1,2,2),dtype=complex)
            if simulation_input['Simulation_Parameters']["differential_absorption"]==True:
                FT_Ewald1=np.zeros((2*order_y+1,2,2),dtype=complex)
                FT_Ewald2=np.zeros((2*order_y+1,2,2),dtype=complex)
            
            for j in range(2):
                for k in range(2):                        
                    FT_Ewald[:,j,k]=np.sum(Phase_Fourier[:,0:-1]*ROI[:,:,j,k],axis=1)
                    if simulation_input['Simulation_Parameters']["differential_absorption"]==True:
                        FT_Ewald1[:,j,k]=np.sum(Phase_Fourier[:,0:-1]*ROI_diff1[:,:,j,k],axis=1)
                        FT_Ewald2[:,j,k]=np.sum(Phase_Fourier[:,0:-1]*ROI_diff2[:,:,j,k],axis=1)    
                         
            output_full=np.zeros((n_det_x,n_det_y,2,2),dtype=complex)          
            output_full1=np.zeros((n_det_x,n_det_y,2,2),dtype=complex)
            output_full2=np.zeros((n_det_x,n_det_y,2,2),dtype=complex) 
            if sample.sim_type=="Crystal":
                
                kernel=np.ones(UC_per_mm_cell,dtype=complex)
                for j in range(UC_per_mm_cell):
                    kernel[j]=self.att_factor**(j/UC_per_mm_cell)
                #the kernel is used to convolve the output from the FT over the micromagnetic cell with an object with unit cell periodicity, in order
                #to generate the equivalent FT from the full object
                 
                Phase_kernel=np.zeros((2*order_y+1,len(kernel)),dtype=complex)
                for j in range(len(kernel)):
                    Phase_kernel[:,j]=np.exp(-2*np.pi*complex(0,1)*j*sample.unit_cell/thickness*qz_pixel_array)
                f_kernel =np.sum(Phase_kernel*kernel,axis=1)
                           
                #applies periodic boundary conditions, expanding a Fourier output to 3X the area
            if simulation_input['Simulation_Parameters']["differential_absorption"]==False:
                self.output=np.zeros((det_z.shape[0],2,2),dtype=complex)
                for j in range(2):
                    for k in range(2):
                        f = FT_Ewald[:,j,k]
                        
                        if sample.sim_type!="Crystal":
                            self.output[:,j,k]= np.reshape(f,(self.output.shape[0])) 
                        if sample.sim_type=="Crystal":
                            self.output[:,j,k]= np.reshape(f*f_kernel,(self.output.shape[0]))
                if output_2D==True:            
                    for j in range(order_x*2+1):
                        
                        if (order_pixels_xz[j] in range(n_det_x)) and (order_pixels_y[j] in range(n_det_y)):
                            output_full[int(order_pixels_xz[j]),int(order_pixels_y[j]),:,:]=self.output[j,:,:]
                                
                    self.output=output_full
                
                
                
                return self.output,self.output  
            
            if simulation_input['Simulation_Parameters']["differential_absorption"]==True:
                self.output1=np.zeros((det_z.shape[0],2,2),dtype=complex)
                self.output2=np.zeros((det_z.shape[0],2,2),dtype=complex)
                
                for j in range(2):
                    for k in range(2):
                        f1 = FT_Ewald1[:,j,k]
                        f2 = FT_Ewald2[:,j,k]
                        
                        if sample.sim_type!="Crystal":
                            self.output1[:,j,k]= np.reshape(f1,(self.output1.shape[0]))
                            self.output2[:,j,k]= np.reshape(f2,(self.output1.shape[0]))
                        if sample.sim_type=="Crystal":
                            self.output1[:,j,k]= np.reshape(f1*f_kernel,(self.output1.shape[0]))
                            self.output2[:,j,k]= np.reshape(f2*f_kernel,(self.output2.shape[0]))
                if output_2D==True:
                    for j in range(order_x*2+1):
                        
                        if (order_pixels_xz[j] in range(n_det_x)) and (order_pixels_y[j] in range(n_det_y)):
                            output_full1[int(order_pixels_xz[j]),int(order_pixels_y[j]),:,:]=self.output1[j,:,:]
                            output_full2[int(order_pixels_xz[j]),int(order_pixels_y[j]),:,:]=self.output2[j,:,:]
                            
                    self.output1=output_full1   
                    self.output2=output_full2
                    
                    
                return self.output1,self.output2

