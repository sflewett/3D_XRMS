# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 10:05:00 2024

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


Co=ud.Atom('Co', mag_amplitude=1, mag_phi=0*u.deg, mag_gamma=90*u.deg)
Pd=ud.Atom('Pd')

amorph_Co = ud.AmorphousLayer('amorph_Fe', 'amorph_Fe', 20*u.nm, 8.9*u.g/u.cm**3,atom=Co,)
amorph_Pd = ud.AmorphousLayer('amorph_Pd', 'amorph_Fe',20*u.nm, 12.0*u.g/u.cm**3,atom=Pd)

multilayer=ud.Structure('Complex_Multilayer')
Pd_thicknesses1=np.linspace(2,2.6,7)
Pd_thicknesses2=2.6
Pd_thicknesses3=np.linspace(2.5,2,6)
n_caps=0
#we must add the number of capping layers outside of the periodic structure, because we need to mark the periodic structure separately in order
#to correctly assign the magnetization distribution

for layers in range(7):
   
    amorph_Co = ud.AmorphousLayer('amorph_Fe', 'amorph_Fe',0.8*u.nm, 8.9*u.g/u.cm**3,atom=Co)
    amorph_Pd = ud.AmorphousLayer('amorph_Pd', 'amorph_Pd',Pd_thicknesses1[layers]*u.nm, 12.0*u.g/u.cm**3,atom=Pd)
    multilayer.add_sub_structure(amorph_Pd,1)
    multilayer.add_sub_structure(amorph_Co,1)
for layers in range(7):
    amorph_Co = ud.AmorphousLayer('amorph_Fe', 'amorph_Fe',0.8*u.nm, 8.9*u.g/u.cm**3,atom=Co)
    amorph_Pd = ud.AmorphousLayer('amorph_Pd', 'amorph_Pd',Pd_thicknesses2*u.nm, 12.0*u.g/u.cm**3,atom=Pd)
    multilayer.add_sub_structure(amorph_Pd,1)
    multilayer.add_sub_structure(amorph_Co,1)
for layers in range(6):
    amorph_Co = ud.AmorphousLayer('amorph_Fe', 'amorph_Fe',0.8*u.nm, 8.9*u.g/u.cm**3,atom=Co)
    amorph_Pd = ud.AmorphousLayer('amorph_Pd', 'amorph_Pd',Pd_thicknesses3[layers]*u.nm, 12.0*u.g/u.cm**3,atom=Pd)
    multilayer.add_sub_structure(amorph_Pd,1)
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
prop_sim["n_caps"]=n_caps
prop_sim["type"]="Multilayer"# Set "Crystal" or "Multilayer"
prop_sim["full_column"]="full"# Set "Crystal" or "Multilayer"
prop_sim['det_size']=[256,256]#detector size in pixels
prop_sim['det_dx']=13.5e-6*4#detector pixel size in metres
prop_sim["det_sample_distance"]=0.15#detector sample distance in metres
prop_sim["periodic_input"]=True
prop_sim["orders_x"]=5
prop_sim["orders_y"]=5
prop_sim["energy"]=780#incident photon energies in eV
prop_sim["angles"]=np.linspace(1,51,201)#incident angles in degrees
prop_sim["pol_in"]=[np.array([[complex(1,0)],[complex(0,1)]]),np.array([[complex(1,0)],[complex(0,-1)]])]#polarization of the incoming light
prop_sim["f_manual_input"]=True
if prop_sim["f_manual_input"]==True:
    prop_sim["f_charge_manual"]=complex(0,60.)#these manual inputs can be later replaced with an energy dependent function interpolating an externally supplied text file
    prop_sim["f_mag_manual"]=complex(0,17.)
#in this example setting we calculate for both left and right circular light
prop_sim["differential_absorption"]=True
#where a field has been applied paralell to the incoming x-rays, then 
#this should be set as true. This is also the case with zero field, but a 
#history of an in-plane field which could be expected to cause aligned Bloch
#walls in the sample. This should be omitted for speed when performing exploratory work, except where
#these differential transmission effects are observed, are expected to be present, and are of importance
#for the study at hand 
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

prop_3D['Mx']="Mx_LSMO.csv"#micromagnetic simulation inputs
prop_3D['My']="My_LSMO.csv"#don´t forget to put these files in the working directory, or add the path to the filename
prop_3D['Mz']="Mz_LSMO.csv"

#There is no obligation here to use micromagnetic simulation outputs, however the output should be a 3D array (maybe with one dimension of size 1)
# and the values can be defined analytically. Prior to saving as a csv file, this should be flattened to 1D using C ordering (for example using the Numpy
#"flatten" command)
prop_3D["dz_average"]=6.0
prop_3D["dz_mag"]=1.5
#average layer thicknesses for the calculation of the differential absorption using a scalar model (activated when "differential absorption" is set to "True")

prop_3D["shape"]=[150,200,20]#array size of the micromagnetic input. PLEASE MAKE SURE YOUR MICROMAGNETIC SIMULTION IS OF THE SAME DIMENSIONS AS YOUR MULTILAYER
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
R,output1,output2,bkg1,bkg2=XRMS_Simulation_Load.Sample2Reflection_Coefficients(sample_multilayer, simulation_input,outputs=["R","output1","output2","bkg1","bkg2"])
#the output files are the diffracted wavefields for each of the two polarization states
#identical in the case where the ["differential_absorption"] is set to false,
#and bkg1 and bkg2 are the background outputs (zero when the background calculation is
#set to false)    
In1=prop_sim["pol_in"][0]
In2=prop_sim["pol_in"][1]
specular1=[]
specular2=[]
output1=np.array(output1)
output2=np.array(output2)
for k in range(output1.shape[0]):
    
    temp1=output1[k,:,:,:,:]
    temp2=output2[k,:,:,:,:]
    Intensity1=(In1[0]*temp1[:,:,0,0]+In1[1]*temp1[:,:,0,1])*np.conj(In1[0]*temp1[:,:,0,0]+In1[1]*temp1[:,:,0,1])+\
    (In1[0]*temp1[:,:,1,0]+In1[1]*temp1[:,:,1,1])*np.conj(In1[0]*temp1[:,:,1,0]+In1[1]*temp1[:,:,1,1])
    
    Intensity2=(In2[0]*temp2[:,:,0,0]+In2[1]*temp2[:,:,0,1])*np.conj(In2[0]*temp2[:,:,0,0]+In2[1]*temp2[:,:,0,1])+\
    (In2[0]*temp2[:,:,1,0]+In2[1]*temp2[:,:,1,1])*np.conj(In2[0]*temp2[:,:,1,0]+In2[1]*temp2[:,:,1,1])
    
    #calculating the outgoing intensities combining the results for the polarizations
    
    beam_stop=np.ones(Intensity1.shape)
    beam_stop[120:135,120:135]=0
    
    #background1=bkg1[k].diffuse_background[1:-1,1:-1]
    #background2=bkg2[k].diffuse_background[1:-1,1:-1]
    
    #IB1=np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(Intensity1[1:-1,1:-1])*np.fft.fft2(background1)))
    #IB2=np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(Intensity2[1:-1,1:-1])*np.fft.fft2(background2)))
    
    #final step of convolving the background with the predicted intensities for a "clean" sample
    
    plt.imshow((np.abs(Intensity1*beam_stop))-(np.abs(Intensity2*beam_stop)))
    plt.show()
    specular1.append(Intensity1[int(prop_sim['det_size'][0]/2),int(prop_sim['det_size'][1]/2)])
    specular2.append(Intensity2[int(prop_sim['det_size'][0]/2),int(prop_sim['det_size'][1]/2)])
    
plt.plot(np.log10(np.array(np.abs(specular1))))
plt.show()
plt.plot(np.log10(np.array(np.abs(specular2))))
plt.show()