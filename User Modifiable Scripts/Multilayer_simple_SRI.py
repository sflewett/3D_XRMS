# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:25:47 2024

@author: sflewett
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 10:05:00 2024

@author: sflewett
"""
#These simulation setup scripts are designed to copy as much as possible of the udkm1Dsim structure
#In addition to defining a sample, it is necessary to include a set of 3 magnetization files
#with the number of z coordinates equal to the number of magnetic layers in the multilayer structure defined in this 
#script. Please see documentation on udkm1Dsim for further details.
import sys
sys.path.append('../Core_Simulation_Classes')
import udkm1Dsim as ud
u = ud.u  # import the pint unit registry from udkm1Dsim
import scipy.constants as constants
import numpy as np
import matplotlib.pyplot as plt
import XRMS_Stepanov_Sinha
import XRMS_Simulation_Load
import background_Lee

#matplotlib inline
u.setup_matplotlib()  # use matplotlib with pint units


Co=ud.Atom('Co', mag_amplitude=1, mag_phi=0*u.deg, mag_gamma=90*u.deg)
Pt=ud.Atom('Pt')
Al=ud.Atom('Al')
Ta=ud.Atom('Ta')
O=ud.Atom('O')

Al2O3 = ud.AtomMixed('Al2O3', id='Al2O3', name='Aluminium0.4_Oxygen0.6')
Al2O3.add_atom(Al, 0.4)
Al2O3.add_atom(O, 0.6)

amorph_Ta = ud.AmorphousLayer('amorph_Ta', 'amorph_Ta', 1.0*u.nm, 16.65*u.g/u.cm**3,atom=Ta)
amorph_Pt = ud.AmorphousLayer('amorph_Pt', 'amorph_Pt', 0.8*u.nm, 21.0*u.g/u.cm**3,atom=Pt)

multilayer=ud.Structure('Simple_Multilayer')


multilayer.add_sub_structure(amorph_Ta,1)
multilayer.add_sub_structure(amorph_Pt,1)
n_caps=2#number of capping layers prior to the periodic structure
for layers in range(20):   
    amorph_Co = ud.AmorphousLayer('amorph_Co', 'amorph_Co',0.8*u.nm, 8.9*u.g/u.cm**3,atom=Co)
    amorph_Al2O3 = ud.AmorphousLayer('amorph_Al2O3', 'amorph_Al2O3',1.0*u.nm, 2.7*u.g/u.cm**3,atom=Al2O3)
    amorph_Pt = ud.AmorphousLayer('amorph_Pt', 'amorph_Pt',1.0*u.nm, 21.0*u.g/u.cm**3,atom=Pt)
    multilayer.add_sub_structure(amorph_Co,1)
    multilayer.add_sub_structure(amorph_Al2O3,1)
    multilayer.add_sub_structure(amorph_Pt,1)

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
prop_sim["det_sample_distance"]=0.10#detector sample distance in metres
prop_sim["orders_x"]=0
prop_sim["orders_y"]=10
prop_sim["energy"]=780#incident photon energies in eV
prop_sim["f_manual_input"]=True
if prop_sim["f_manual_input"]==True:
    prop_sim["f_charge_manual"]=complex(0,60.)#these manual inputs can be later replaced with an energy dependent function interpolating an externally supplied text file
    prop_sim["f_mag_manual"]=complex(0,17.)
prop_sim["angles"]=np.linspace(10,20,101)#incident angles in degrees
prop_sim["pol_in"]=[np.array([[complex(1,0)],[complex(0,1)]]),np.array([[complex(1,0)],[complex(0,-1)]])]#polarization of the incoming light
#in this example setting we calculate for both left and right circular light
prop_sim["differential_absorption"]=True
prop_sim["rotated_stripes"]=True
prop_sim["phi_rotations"]=[90]#phi rotation in degrees
prop_sim["matrix_stack_coordinates"]=[0,32]
#where a field has been applied paralell to the incoming x-rays, then 
#this should be set as true. This is also the case with zero field, but a 
#history of an in-plane field which could be expected to cause aligned Bloch
#walls in the sample. This should be omitted for speed when performing exploratory work, except where
#these differential transmission effects are observed, are expected to be present, and are of importance
#for the study at hand 
prop_sim["extra_absorption"]=3.5
#this is a correction for the Henke tables being used

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

prop_3D['Mx']="../Magnetization Files/SRI_x.csv"#micromagnetic simulation inputs
prop_3D['My']="../Magnetization Files/SRI_y.csv"#don´t forget to put these files in the working directory, or add the path to the filename
prop_3D['Mz']="../Magnetization Files/SRI_z.csv"

#There is no obligation here to use micromagnetic simulation outputs, however the output should be a 3D array (maybe with one dimension of size 1)
# and the values can be defined analytically. Prior to saving as a csv file, this should be flattened to 1D using C ordering (for example using the Numpy
#"flatten" command)

prop_3D["shape"]=[2,384,20]#array size of the micromagnetic input. PLEASE MAKE SURE YOUR MICROMAGNETIC SIMULTION IS OF THE SAME DIMENSIONS AS YOUR MULTILAYER
#the number of magnetic layers must be the same as the z dimension
prop_3D["x_y_pixel_sizes"]=[2.,2.]#pixel sizes in nm. for crystals this need not be equal to the unit cell parameters, however
#for multilayers the z pixel sizes need to match the real multilayer structure, or be a scaler representing the periodicity
prop_3D["dz_average"]=2.8
prop_3D["dz_mag"]=0.8

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

R,output1,output2,bkg1,bkg2=XRMS_Simulation_Load.Sample2Reflection_Coefficients(simulation_input,outputs=["output1","output2","bkg1","bkg2"])
#the output files are the diffracted wavefields for each of the two polarization states
#identical in the case where the ["differential_absorption"] is set to false,
#and bkg1 and bkg2 are the background outputs (zero when the background calculation is
#set to false)    
In1=np.conj(prop_sim["pol_in"][0])
In2=np.conj(prop_sim["pol_in"][1])
specular1=[]
specular2=[]
output1=np.array(output1)
output2=np.array(output2)
Intensities_sum=np.zeros((output1.shape[0],output1.shape[1]))
Intensities_diff=np.zeros((output1.shape[0],output1.shape[1]))
for k in range(output1.shape[0]):
    
    temp1=output2[k,:,:,:]
    temp2=output1[k,:,:,:]
    Intensity1=(In1[0]*temp1[:,0,0]+In1[1]*temp1[:,0,1])*np.conj(In1[0]*temp1[:,0,0]+In1[1]*temp1[:,0,1])+\
    (In1[0]*temp1[:,1,0]+In1[1]*temp1[:,1,1])*np.conj(In1[0]*temp1[:,1,0]+In1[1]*temp1[:,1,1])
    
    Intensity2=(In2[0]*temp2[:,0,0]+In2[1]*temp2[:,0,1])*np.conj(In2[0]*temp2[:,0,0]+In2[1]*temp2[:,0,1])+\
    (In2[0]*temp2[:,1,0]+In2[1]*temp2[:,1,1])*np.conj(In2[0]*temp2[:,1,0]+In2[1]*temp2[:,1,1])
    
    #calculating the outgoing intensities combining the results for the polarizations
    
    beam_stop=np.ones(Intensity1.shape)
    beam_stop[prop_sim["orders_y"]]=0
    
    #background1=bkg1[k].diffuse_background[1:-1,1:-1]
    #background2=bkg2[k].diffuse_background[1:-1,1:-1]
    
    #IB1=np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(Intensity1[1:-1,1:-1])*np.fft.fft2(background1)))quit()

    #IB2=np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(Intensity2[1:-1,1:-1])*np.fft.fft2(background2)))
    
    #final step of convolving the background with the predicted intensities for a "clean" sample
    
    Intensities_sum[k,:]=((np.abs(Intensity1*beam_stop))+(np.abs(Intensity2*beam_stop)))
    
    Intensities_diff[k,:]=((np.abs(Intensity1*beam_stop))-(np.abs(Intensity2*beam_stop)))/((np.abs(Intensity1*beam_stop))+(np.abs(Intensity2*beam_stop))+1e-6)
    Intensities_diff[0,0]=1.0
    Intensities_diff[-1,-1]=-1.0
    
   
    specular1.append(Intensity1[int(prop_sim["orders_y"])])
    specular2.append(Intensity2[int(prop_sim["orders_y"])])


for i in range(output1.shape[0]):
    fig, (ax1, ax2) = plt.subplots(1,2)
    filename="../figures/test"+str(i)
    im = ax1.plot(Intensities_sum[i,:], animated=True)
    #plt.savefig(filename+"sum.png")
    im = ax2.plot(Intensities_diff[i,:], animated=True)
    plt.savefig(filename+"sumdiff.png")
# fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,15))
# extent=[0,20,10,40]
# im=ax1.imshow(np.log(np.abs(Intensities_sum)),extent=extent)
# xticks=[1,4,7,10,13,16,19]
# xticklabels=["-3","-2","-1","0","1","2","3"]
# yticks=list(np.linspace(10,40,31))#[10,11,12,13,14,15,16,17,18,19,20]
# yticklabels=[]
# for j in range(31):
#     yticklabels.append(str(40-j))
# ax1.set_title("Sum of both circular polarizations \n (log colourscale)",fontsize=12)
# ax1.set_xlabel("Diffraction_Order",fontsize=10)
# ax1.set_ylabel("Angle of Incidence (deg)",fontsize=10)
# ax1.set(xticks=xticks,yticks=yticks)
# ax1.set_xticklabels(xticklabels,fontsize=8)
# ax1.set_yticklabels(yticklabels,fontsize=8)

# im=ax2.imshow(np.abs(Intensities_diff)**1.0*np.sign(Intensities_diff),extent=extent)
# ax2.set_title("Normalized difference of both circular polarizations \n (lin colourscale)",fontsize=12)
# ax2.set_xlabel("Diffraction_Order",fontsize=10)
# ax2.set_ylabel("Angle of Incidence (deg)",fontsize=10)
# ax2.set(xticks=xticks,yticks=yticks)
# ax2.set_xticklabels(xticklabels,fontsize=8)
# ax2.set_yticklabels(yticklabels,fontsize=8)

# filename="../figures/allangles"
# plt.savefig(filename+"sumdiff.png")
