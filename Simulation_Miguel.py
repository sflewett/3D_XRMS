# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:11:36 2024

@author: sflewett
"""

#These simulation setup scripts are designed to copy as much as possible of the udkm1Dsim structure
#In addition to defining a sample, it is necessary to include a set of 3 magnetization files
#with the number of z coordinates equal to the number of magnetic layers in the multilayer structure defined in this 
#script. Please see documentation on udkm1Dsim for further details.
import udkm1Dsim as ud
u = ud.u  # import the pint unit registry from udkm1Dsim
import scipy.constants as constants
import numpy as np
import matplotlib.pyplot as plt
import XRMS_Simulation_Load
import background_Lee

#matplotlib inline
u.setup_matplotlib()  # use matplotlib with pint units


Fe=ud.Atom('Fe', mag_amplitude=1, mag_phi=0*u.deg, mag_gamma=90*u.deg)
Co=ud.Atom('Pd')

multilayer=ud.Structure('Miguel_structure')
Fe_thickness=500.0
Co_thickness=2.6




amorph_Fe = ud.AmorphousLayer('amorph_Fe', 'amorph_Fe',Fe_thickness*u.nm, 8.9*u.g/u.cm**3,atom=Fe)
amorph_Co = ud.AmorphousLayer('amorph_Co', 'amorph_Co',Co_thickness*u.nm, 8.9*u.g/u.cm**3,atom=Co)
multilayer.add_sub_structure(amorph_Fe,1)
multilayer.add_sub_structure(amorph_Co,1)
#in this example, we use an irregularly spaced multilayer, of the same kind as the example in the associated publication    
print(multilayer)

multilayer.visualize()
# =============================================================================
# #Up to this point, we have defined the chemical layer structure. The following definitions
# #are for the 3D magnetic structure from micromagnetic simulations
# =============================================================================

# =============================================================================
# #The next section is to load up the simulated experimental parameters
# =============================================================================
prop_sim={}
prop_sim["type"]="Multilayer"# Set "Crystal" or "Multilayer"
prop_sim["full_column"]="full"# Set "Crystal" or "Multilayer"
prop_sim['det_size']=[256,256]#detector size in pixels
prop_sim['det_dx']=13.5e-6#detector pixel size in metres
prop_sim["det_sample_distance"]=0.02#detector sample distance in metres
prop_sim["periodic_input"]=True
prop_sim["orders_x"]=50
prop_sim["orders_y"]=50
prop_sim["energy"]=708#incident photon energies in eV
prop_sim["angles"]=np.linspace(20,20,1)#incident angles in degrees
prop_sim["pol_in"]=[np.array([[complex(1,0)],[complex(0,1)]]),np.array([[complex(1,0)],[complex(0,-1)]])]#polarization of the incoming light
#in this example setting we calculate for both left and right circular light
prop_sim["differential_absorption"]=False
#where a field has been applied paralell to the incoming x-rays, then 
#this should be set as true. This is also the case with zero field, but a 
#history of an in-plane field which could be expected to cause aligned Bloch
#walls in the sample. This should be omitted for speed when performing exploratory work, except where
#these differential transmission effects are observed, are expected to be present, and are of importance
#for the study at hand 
prop_sim["periodic_input"]=True

prop_sim["calculate_background"]=False
#when set to True, this calculates the diffuse background scatter, and is especially important when working with
#rougher samples with disordered domain structures. For stripe domains, this can often be omitted for speed.
#where this parameter is set to False, it is not necessary to update the background parameters as follows
prop_sim["specular_roughness"]=0e-10
#set to a non zero value for a specular calculation, or set to zero for the 3D calculation.
#For the 3D calculation, the roughness is introduced via a convolution with the background model
prop_sim["TMOKE_Threshold"]=0.999
#when the Stepanov-Sinha scattering formalism is used for calculating the reflection and transmission coefficients,
#it is necessary to define a transverse magnetization threshold above which the calculation is performed
#using an alternative calculation scheme. (section III C of the original Stepanov Sinha paper) 
#This is because the general form has a matrix singularity in the
#case of the magnetization vector being perpencicular to the scattering plane (Transverse MOKE)
# =============================================================================
# =============================================================================
#load up 3D simulation parameters
prop_3D={}

Mfile=np.load("achiral_pattern.npy")

temp=np.zeros((1024,1024,1))
Intensity1=np.zeros((prop_sim['det_size'][0],prop_sim['det_size'][1]))
Intensity2=np.zeros((prop_sim['det_size'][0],prop_sim['det_size'][1]))
for ver in range(4):
    for hor in range(4):

        temp[:,:,0]=Mfile[0,1024*(ver):1024*(ver+1),1024*(hor):1024*(hor+1)]
        #temp[:,:,1]=Mfile[0,0:1024,0:1024]-Mfile[0,0:1024,0:1024]
        np.savetxt("Mx.csv", np.ndarray.flatten(np.transpose(temp)), delimiter=",")
        temp=np.zeros((1024,1024,1))
        temp[:,:,0]=Mfile[1,1024*(ver):1024*(ver+1),1024*(hor):1024*(hor+1)]
        #temp[:,:,1]=Mfile[1,0:1024,0:1024]-Mfile[1,0:1024,0:1024]
        np.savetxt("My.csv", np.ndarray.flatten(np.transpose(temp)), delimiter=",")
        temp=np.zeros((1024,1024,1))
        temp[:,:,0]=Mfile[2,1024*(ver):1024*(ver+1),1024*(hor):1024*(hor+1)]
        #temp[:,:,1]=Mfile[2,0:1024,0:1024]-Mfile[2,0:1024,0:1024]
        np.savetxt("Mz.csv", np.ndarray.flatten(temp), delimiter=",")
        prop_3D['Mx']="Mx.csv"#micromagnetic simulation inputs
        prop_3D['My']="My.csv"#don´t forget to put these files in the working directory, or add the path to the filename
        prop_3D['Mz']="Mz.csv"
        
        #There is no obligation here to use micromagnetic simulation outputs, however the output should be a 3D array (maybe with one dimension of size 1)
        # and the values can be defined analytically. Prior to saving as a csv file, this should be flattened to 1D using C ordering (for example using the Numpy
        #"flatten" command)
        
        prop_3D["shape"]=[1024,1024,1]#array size of the micromagnetic input. PLEASE MAKE SURE YOUR MICROMAGNETIC SIMULTION IS OF THE SAME DIMENSIONS AS YOUR MULTILAYER
        #the number of magnetic layers must be the same as the z dimension
        prop_3D["x_y_pixel_sizes"]=[2.,2.]#pixel sizes in nm. for crystals this need not be equal to the unit cell parameters, however
        #for multilayers the z pixel sizes need to match the real multilayer structure, or be a scaler representing the periodicity
        
        #f_Mag=[]
        #f_Mag.append((get_f0(energy,f_datafile),get_f1(energy,f_datafile),get_f2(energy,f_datafile)))
        #f_Mag.append((get_f0(energy,f_datafile),get_f1(energy,f_datafile),get_f2(energy,f_datafile)))
        #where these values are provided, they override those given in the Henke or Chantler tables. 
        
        #for multiple energies, the reference atomic scattering factors need to be saved, and accessed via functions. 
        
        
        # =============================================================================
        # #This final section is to load up the diffuse background simulation parameters
        # =============================================================================
        bkg_params={}
        bkg_params["sigmas"] = np.array([4.6e-10,4.6e-10])#charge and magnetic roughness
        bkg_params["eta_par"] = np.array([2.0e-8,2.0e-8])#in plane correlation lengths (charge and magnetic)
        bkg_params["h"] = np.array([0.3,0.3])#roughness exponents (charge and magnetic)  
        bkg_params["eta_perp"]=4.5e-8#out of plane correlation length
        #Please refer to the Lee 2003 PRB for details about these parameters
        simulation_input={}
        
        simulation_input['sample']=multilayer
        simulation_input['3D_Sample_Parameters']=prop_3D
        simulation_input['Simulation_Parameters']=prop_sim
        simulation_input['Background_Parameters']=bkg_params
        
        sample_multilayer=XRMS_Simulation_Load.Generic_sample(simulation_input)
        R,output1,output2,bkg1,bkg2=XRMS_Simulation_Load.Sample2Reflection_Coefficients(sample_multilayer, simulation_input)
        #the output files are the diffracted wavefields for each of the two polarization states
        #identical in the case where the ["differential_absorption"] is set to false,
        #and bkg1 and bkg2 are the background outputs (zero when the background calculation is
        #set to false)    
        In1=prop_sim["pol_in"][0]
        In2=prop_sim["pol_in"][1]
        #specular1=[]
        #specular2=[]
        print(hor)
        print(ver)
   
    
        temp1=output1
        temp2=output2
        Intensity1=Intensity1+(In1[0]*temp1[:,:,0,0]+In1[1]*temp1[:,:,0,1])*np.conj(In1[0]*temp1[:,:,0,0]+In1[1]*temp1[:,:,0,1])+\
        (In1[0]*temp1[:,:,1,0]+In1[1]*temp1[:,:,1,1])*np.conj(In1[0]*temp1[:,:,1,0]+In1[1]*temp1[:,:,1,1])
        
        Intensity2=Intensity2+(In2[0]*temp2[:,:,0,0]+In2[1]*temp2[:,:,0,1])*np.conj(In2[0]*temp2[:,:,0,0]+In2[1]*temp2[:,:,0,1])+\
        (In2[0]*temp2[:,:,1,0]+In2[1]*temp2[:,:,1,1])*np.conj(In2[0]*temp2[:,:,1,0]+In2[1]*temp2[:,:,1,1])

#calculating the outgoing intensities combining the results for the polarizations

beam_stop=np.ones(Intensity1.shape)
beam_stop[128,128]=0
  

            
plt.imshow((np.abs(Intensity1*beam_stop))-(np.abs(Intensity2*beam_stop)))
plt.show()
