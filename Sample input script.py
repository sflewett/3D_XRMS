import udkm1Dsim as ud
u = ud.u  # import the pint unit registry from udkm1Dsim
import scipy.constants as constants
import numpy as np
import matplotlib.pyplot as plt
import XRMS_Simulation_Load
from XRMS_Stepanov_Sinha import XRMS_Simulate
import background_Lee
#matplotlib inline

#These simulation setup scripts are designed to copy as much as possible of the udkm1Dsim structure
#In addition to defining a sample, it is necessary to include a set of 3 magnetization files
#with the number of z coordinates equal the desired magnetic resolution of the simulation.
#Reflection coefficients will be calculated at this resolution, however when the calculation of the
#final diffraction is made, a convolution will be made to increase the resolution to the length
#of the crystal C-axis

#Please see documentation on udkm1Dsim for further details.



u.setup_matplotlib()  # use matplotlib with pint units

O = ud.Atom('O')
Sr = ud.Atom('Sr')
Mn = ud.Atom('Mn', mag_amplitude=1, mag_phi=0*u.deg, mag_gamma=90*u.deg)#atomic_form_factor_path=YOUR FORM FACTOR PATH, also MAGNETIC FORM FACTOR PATH
La= ud.Atom('La')
Si= ud.Atom('Si')
Fe= ud.Atom('Fe')
Ni= ud.Atom('Ni')

LS = ud.AtomMixed('LS', id='LS', name='Lanthanum0.6666_Strontium0.3333')
LSMO_bulk=ud.AtomMixed('LSMO_bulk', id='LSMO_bulk', name='LSMO_bulk')
amorph_LSMO = ud.AmorphousLayer('amorph_LSMO_bulk', 'amorph_LSMO_bulk',1.0*u.nm, 6.5*u.g/u.cm**3,atom=LSMO_bulk)

Permalloy_atom=ud.AtomMixed('Permalloy', id='Permalloy', name='Iron0.21_Nickel0.79')
LS.add_atom(La, 0.6666)
LS.add_atom(Sr, 0.3333)
Permalloy_atom.add_atom(Fe, 0.21)
Permalloy_atom.add_atom(Ni, 0.79)
LSMO_bulk.add_atom(La,2/15.0)
LSMO_bulk.add_atom(Sr,1/15.0)
LSMO_bulk.add_atom(Mn,3/15.0)
LSMO_bulk.add_atom(O,0.6)
#full compound to get bulk atomic scattering factor

n_caps=0
#we must add the number of capping layers outside of the periodic structure, because we need to mark the periodic structure separately in order
#to correctly assign the magnetization distribution
# c-axis lattice constants of the two layers
c_Si_sub = 3.905*u.angstrom
c_LSMO = 3.88*u.angstrom
c_Permalloy=3.5*u.angstrom
# LSMO layer
prop_LSMO = {}
prop_LSMO['a_axis'] = c_LSMO  # a-Axis
prop_LSMO['b_axis'] = c_LSMO  # b-Axis

prop_Permalloy = {}
prop_Permalloy['a_axis'] = c_Permalloy  # a-Axis
prop_Permalloy['b_axis'] = c_Permalloy  # b-Axis


LSMO = ud.UnitCell('LSMO', 'LSMO', c_LSMO, **prop_LSMO)
LSMO.add_atom(O, 0)
LSMO.add_atom(O, 0)
LSMO.add_atom(Mn, 0)

LSMO.add_atom(O, 0.5)
LSMO.add_atom(LS, 0.5)

prop_Si_sub = {}
prop_Si_sub['a_axis'] = c_Si_sub  # a-Axis
prop_Si_sub['b_axis'] = c_Si_sub  # b-Axis
prop_Si_sub['deb_Wal_Fac'] = 0  # Debye-Waller factor


Si_sub = ud.UnitCell('Si_sub', 'Silicon Substrate',
                      c_Si_sub, **prop_Si_sub)
Si_sub.add_atom(O, 0)
Si_sub.add_atom(Si, 0)
Si_sub.add_atom(O, 0.5)
Si_sub.add_atom(O, 0.5)
Si_sub.add_atom(Si, 0.5)

Permalloy = ud.UnitCell('Permalloy', 'Permalloy', c_Permalloy, **prop_Permalloy)
Permalloy.add_atom(Permalloy_atom,0)
Permalloy.add_atom(Permalloy_atom,0)
Permalloy.add_atom(Permalloy_atom,0.5)
Permalloy.add_atom(Permalloy_atom,0.5)


S = ud.Structure('100nm_LSMO')
n_repeats_LSMO=np.int64(np.round(93.1e-9/c_LSMO.magnitude*1e10))
n_repeats_Permalloy=np.int64(np.round(100e-9/c_Permalloy.magnitude*1e10))
S.add_sub_structure(Permalloy, n_repeats_Permalloy)
S.add_sub_structure(LSMO, n_repeats_LSMO)  # add 100 layers of SRO to sample
S.add_sub_structure(Si_sub, 1000)  # add 1000 layers of Si substrate

print(S)

S.visualize()
# =============================================================================
# #Up to this point, we have defined the chemical layer structure. The following definitions
# #are for the 3D magnetic structure from micromagnetic simulations
# =============================================================================

# =============================================================================
# #The next section is to load up the simulated experimental parameters
# =============================================================================
prop_sim={}
prop_sim["n_caps"]=n_caps
prop_sim['det_size']=[256,256]#detector size in pixels
prop_sim['det_dx']=13.5e-6*4#detector pixel size in metres
prop_sim["det_sample_distance"]=0.15#detector sample distance in metres
prop_sim["energy"]=640#incident photon energies in eV
prop_sim["angles"]=np.linspace(10,20,2)#incident angles in degrees
prop_sim["pol_in"]=[[[complex(1.0,0.0)],[complex(0.0,1.0)]],[[complex(1.0,0.0)],[complex(0.0,-1.0)]]]#polarization of the incoming light
#in this example setting we calculate for both left and right circular light
prop_sim["type"]="Crystal"# Set "Crystal" or "Multilayer"
if prop_sim["type"]=="Crystal":
    prop_sim["f_bulk"]=amorph_LSMO._atom.get_atomic_form_factor(prop_sim["energy"])
    prop_sim["extra_absorption"]=2
#choose here whether the sample is of a crystal or multilayer step. From a software point of view, for a crystal, there is an interpolation between the calculation
#of the reflection coefficients and the calculation of the resulting diffraction pattern. For the "Multilayer" setting, the micromagnetic simulation must have the same
#number of layers as the structure defined in this present script. Where this option is chosen, we may make the approximation that the magnetization can be expressed in blocks of many unit cells
#which forces us to approximate the attenuation using a simple scaler model (to avoid artificial "Bragg" interference over the micromagnetic cell lattice). This 
#additional absorption can be set via the "extra_absorption" keyword.

prop_sim["differential_absorption"]=True
prop_sim["orders_x"]=10
prop_sim["orders_y"]=10

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
# =============================================================================
# =============================================================================
#load up 3D simulation parameters
prop_3D={}

prop_3D['Mx']="Mx_LSMO.csv"#micromagnetic simulation inputs
prop_3D['My']="My_LSMO.csv"#don´t forget to put these files in the working directory, or add the path to the filename
prop_3D['Mz']="Mz_LSMO.csv"


#The X-Y-Z grid on which each micromagnetic simulation block is entered must be equal.

#There is no obligation here to use micromagnetic simulation outputs, however the output should be a 3D array (maybe with one dimension of size 1)
# and the values can be defined analytically. Prior to saving as a csv file, this should be flattened to 1D using C ordering (for example using the Numpy
#"flatten" command)
prop_3D["substructure_mag"]="LSMO"#Substructure for which the micromagnetic simulations correspond. 

prop_3D["shape"]=[150,200,20]#array size of the micromagnetic input
prop_3D["x_y_pixel_sizes"]=[5.,5.]#pixel sizes in nm. for crystals this need not be equal to the unit cell parameters, however
#for multilayers the z pixel sizes need to match the real multilayer structure, or be a scaler representing the periodicity
prop_3D["z_pixel_sizes"]=0.388*12
prop_3D["total_thickness"]=0.388*12*prop_3D["shape"][2]*1e-9

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

simulation_input['sample']=S
simulation_input['3D_Sample_Parameters']=prop_3D
simulation_input['Simulation_Parameters']=prop_sim
simulation_input['Background_Parameters']=bkg_params

sample_crystal=XRMS_Simulation_Load.Generic_sample(simulation_input)

R,output1,output2,bkg1,bkg2=XRMS_Simulation_Load.Sample2Reflection_Coefficients(sample_crystal, simulation_input,outputs=["R","output1","output2","bkg1","bkg2"])
#bkg1,bkg2=background_Lee.background_main(simulation_input,theta=25*np.pi/180) 
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
    beam_stop[128,128]=0
    
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