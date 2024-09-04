# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 13:41:01 2024

@author: sflewett
"""
import sys
sys.path.append('../')

import numpy as np
from XRMS_Stepanov_Sinha import XRMS_Simulate
import csv
import background_Lee

class Generic_sample():
    
    #the Generic_sample class is the link between the sample input according to the format of Daniel Schick udk
    
    def __init__(self,simulation_input,incident_beam_index=0):
        
        S=simulation_input["sample"]
        params_3D=simulation_input['3D_Sample_Parameters']
        params_general=simulation_input['Simulation_Parameters']
        sim_type=params_general["type"]
        self.sim_type=sim_type
        self.Incident=np.array(simulation_input['Simulation_Parameters']["pol_in"][incident_beam_index])
        if "TMOKE_Threshold" in params_general:
            self.TMOKE_Threshold=params_general["TMOKE_Threshold"]
        else:
            self.TMOKE_Threshold=0.995
        if "phi_rotate" in params_general:
            self.phi_rotate=params_general["phi_rotate"]
        else:
            self.phi_rotate=0.0
        if "full_column" in params_general:
            self.full_column=params_general["full_column"]
        else:
            self.full_column="full"
        if "matrix_stack_coordinates" in params_general:
            self.stack_coordinates=params_general["matrix_stack_coordinates"]
        else:
            self.stack_coordinates=[0,0]
        self.energy=params_general["energy"]
        self.dx=params_3D["x_y_pixel_sizes"][0]
        self.dy=params_3D["x_y_pixel_sizes"][1]
        micromag_size=params_3D["shape"]
        self.size_x=micromag_size[0]
        self.size_y=micromag_size[1]
        layer_list=["vacuum"]
        
        if "f_manual_input" in params_general and params_general["f_manual_input"]==True:
            self.f_charge_manual=params_general["f_charge_manual"]
            self.f_mag_manual=params_general["f_mag_manual"]
            
        if sim_type=="Crystal":
            n_repeats=[]
            unit_cells=[]
            thicknesses=[]
            self.f_bulk=params_general["f_bulk"]
            
            for j in range(S.get_number_of_unique_layers()):
                temp=S.sub_structures[j]
                n_repeats.append(temp[1])
                unit_cells.append(S.sub_structures[j][0].c_axis.magnitude*1e-9)
                thicknesses.append(unit_cells[j]*n_repeats[j])
            z_positions=[]
            
            for j in range(len(S.sub_structures)):
                layer_list.append(S.sub_structures[j][0].id)       
            
            self.unit_cells=unit_cells
            self.thicknesses=thicknesses
            self.n_repeats=n_repeats
            
        if sim_type=="Multilayer":
            
            n_caps=simulation_input['Simulation_Parameters']["n_caps"]
            thicknesses=[]
            z_positions=[0.]
            for j in range(len(S.sub_structures)):
                thicknesses.append(S.sub_structures[j][0].thickness.magnitude)
                if j==0:
                    z_positions.append(np.double(S.sub_structures[j][0].thickness.magnitude))
                else:
                    z_positions.append(np.double(z_positions[j]+S.sub_structures[j][0].thickness.magnitude))
            for j in range(len(S.sub_structures)):
                layer_list.append(S.sub_structures[j][0].id)       
            
            self.thicknesses=np.array(thicknesses)*1e-9
            self.z=np.array(z_positions)*1e-9 
      
        self.M=[]
                
         #in the next block of code we set up the 3D array on which reflection coefficients are to be calculated
        if sim_type=="Crystal":
            self.size_z=params_3D["shape"][2]#number of layers in the micromagnetic simulation
            index=layer_list.index(params_3D["substructure_mag"])
            self.unit_cell=self.unit_cells[index-1]
            layer_magnetized=[]
            
            total_magnetization=0
            for j in range(len(S.sub_structures[index-1][0].atoms)):
                total_magnetization=total_magnetization+S.sub_structures[index-1][0].atoms[j][0].mag_amplitude
                z_positions.append(np.double(S.sub_structures[index-1][0].atoms[j][2]))
                #the -1 is to account for the vacuum layer which isn´t defined in the user interface script
            if total_magnetization==0:
                layer_magnetized.append(False)
            else:
                layer_magnetized.append(True)
                    
            unique = list(set(z_positions))
            n_unique=len(unique)
            self.n_unique=n_unique    
            
            self.dz=params_3D["z_pixel_sizes"]
            micromag_files=[params_3D["Mx"],params_3D["My"],params_3D["Mz"]]#order is longitudinal, transverse, polar
            M_temp=(np.zeros((3,micromag_size[0],micromag_size[1],micromag_size[2])))
            for j in range(3):
                with open(micromag_files[j]) as f:
                    reader = csv.reader(f)
                    data = list(reader)
                    data=np.array(data)
                    data_reshaped=data.reshape(micromag_size,order="C")
                    M_temp[j,:,:,:]=data_reshaped
            if (micromag_size[0]==1 and micromag_size[1]==1) or self.full_column=="column":
                self.sigma_roughness=params_3D["specular_roughness"]
            else:
                self.sigma_roughness=0.
                
            
            
            M2=np.zeros((3,self.size_x,self.size_y,self.size_z*n_unique+1),dtype=complex)
            na=np.zeros((self.size_x,self.size_y,self.size_z*n_unique+1),dtype=complex)
            f_Charge=np.zeros((self.size_x,self.size_y,self.size_z*n_unique+1),dtype=complex)
            f_Mag=np.zeros((self.size_x,self.size_y,self.size_z*n_unique+1),dtype=complex)
            f_Mag2=np.zeros((self.size_x,self.size_y,self.size_z*n_unique+1),dtype=complex)
            z=[]
            maglayer_count=0
            for l in range(self.size_z*n_unique+1):
                if l==0 and layer_list[index-1]=="vacuum":
                    M2[:,:,:,l]=np.zeros((3,self.size_x,self.size_y))
                    na[:,:,l]=1.
                    f_Charge[:,:,l]=complex(1e-6,1e-4)
                    f_Mag[:,:,l]=complex(0,0)
                    f_Mag2[:,:,l]=complex(0,0)
                    z.append(-self.dz/2)
                    #we add the vacuum layer in the sample class
                if l==0 and layer_list[index-1]!="vacuum":
                    M2[:,:,:,l]=np.zeros((3,self.size_x,self.size_y))
                    na[:,:,l]=(S.sub_structures[index-2][0].num_atoms/S.sub_structures[index-2][0].volume).magnitude
                    #index includes the vacuum layer, so starts at 1
                    temp=S.sub_structures[index-2][0].get_atom_positions()
                    temp2=np.array(list(range(len(temp))))
                    indices=temp2[temp==max(temp)]
                    f_Charge[:,:,l]=0
                    for kk in (indices):
                        temp=S.sub_structures[index-2][0].atoms[kk][0].get_atomic_form_factor(self.energy)
                        if np.imag(temp)<0:
                            temp=np.conj(temp)
                        f_Charge[:,:,l]=temp               
                    f_Mag[:,:,l]=complex(0,0)
                    f_Mag2[:,:,l]=complex(0,0)
                    z.append(-self.dz/2)#initial vacuum layer above the sample
                if l>0:
                    l2=(l-1)//n_unique
                    l3=(l-1)%n_unique
                    f_Charge[:,:,l]=complex(0,0)
                    f_Mag[:,:,l]=complex(0,0)
                    f_Mag2[:,:,l]=complex(0,0)
                    temp=S.sub_structures[index-1][0].get_atom_positions()
                    temp_unique = list(set(z_positions))
                    temp2=np.array(list(range(len(temp))))
                    indices=temp2[temp==temp_unique[l%n_unique]]
                    na[:,:,l]+=(S.sub_structures[index-1][0].num_atoms/S.sub_structures[index-1][0].volume).magnitude
                    z.append(self.dz*(l2)+temp_unique[l3]*self.unit_cell*1e9)
                    for kk in (indices):
                        temp=S.sub_structures[index-1][0].atoms[kk][0].get_atomic_form_factor(self.energy)
                        if np.imag(temp)<0:
                            temp=np.conj(temp)
                        
                        temp2=S.sub_structures[index-1][0].atoms[kk][0].get_magnetic_form_factor(self.energy)/len(indices)
                        
                        if "f_manual_input" not in params_general or params_general["f_manual_input"]==False or temp2==0:
                            f_Charge[:,:,l]=temp
                            f_Mag[:,:,l]+=S.sub_structures[index-1][0].atoms[kk][0].get_magnetic_form_factor(self.energy)/len(indices)
                            f_Mag2[:,:,l]=complex(0,0)
                        
                        if temp2!=0 and "f_manual_input" in params_general and params_general["f_manual_input"]==True:
                            f_Charge[:,:,l]=params_general["f_charge_manual"]#/len(indices)
                            f_Mag[:,:,l]+=params_general["f_mag_manual"]/len(indices)
                            
                    if f_Mag[:,:,l].sum()!=0:
                        maglayer_count+=1
                        M2[:,:,:,l]=M_temp[:,:,:,l2]
                        f_Charge_maglayer_scalar=f_Charge[0,0,l]
                        f_Mag_scalar=f_Mag[0,0,l]
                        na_scalar=na[0,0,l]
            self.dz_mag=maglayer_count/(self.size_z*n_unique)*self.dz*1e-9        
            self.n_unique=n_unique
            M3=M2.copy()
            M3[0,:,:,:]=M2[0,:,:,:]*np.cos(self.phi_rotate*np.pi/180)+M2[1,:,:,:]*np.sin(self.phi_rotate*np.pi/180)
            M3[1,:,:,:]=-M2[0,:,:,:]*np.sin(self.phi_rotate*np.pi/180)+M2[1,:,:,:]*np.cos(self.phi_rotate*np.pi/180)
            self.M=(M3)
            self.na=(na)
            self.f_Charge=f_Charge
            self.f_Mag=f_Mag
            self.f_Mag2=f_Mag2
            self.z=np.array(z,dtype=np.double)*1e-9         
            self.f_Charge_maglayer_scalar=f_Charge_maglayer_scalar
            self.f_Mag_scalar=f_Mag_scalar  
            self.na_scalar=na_scalar
        if sim_type=="Multilayer":
            
            micromag_files=[params_3D["Mx"],params_3D["My"],params_3D["Mz"]]#order is longitudinal, transverse, polar
            M_temp=(np.zeros((3,micromag_size[0],micromag_size[1],micromag_size[2])))
            for j in range(3):
                with open(micromag_files[j]) as f:
                    reader = csv.reader(f)
                    data = list(reader)
                    data=np.array(data)
                    data_reshaped=data.reshape(micromag_size,order="C")
                    M_temp[j,:,:,:]=data_reshaped
            if (micromag_size[0]==1 and micromag_size[1]==1) or self.full_column=="column":
                self.sigma_roughness=params_3D["specular_roughness"]
            else:
                self.sigma_roughness=0.
            
            
            self.size_z=len(self.z)
            M2=np.zeros((3,self.size_x,self.size_y,self.size_z),dtype=complex)
            na=np.zeros((self.size_x,self.size_y,self.size_z),dtype=complex)
            f_Charge=np.zeros((self.size_x,self.size_y,self.size_z),dtype=complex)
            f_Mag=np.zeros((self.size_x,self.size_y,self.size_z),dtype=complex)
            f_Mag2=np.zeros((self.size_x,self.size_y,self.size_z),dtype=complex)            
            mag_counter=0#counting the number of magnetic layers
            l2=(len(z_positions)-1)//M_temp.shape[-1]
            for l in range(self.size_z):
                
                if l==0:
                    M2[:,:,:,l]=np.zeros((3,self.size_x,self.size_y))
                    na[:,:,l]=1.
                    f_Charge[:,:,l]=complex(1e-9,1e-9)#small number so as not to generate singularities
                    f_Mag[:,:,l]=complex(0,0)
                    f_Mag2[:,:,l]=complex(0,0)
                    #we add the vacuum layer in the sample class
                
                if l>0:
                    if S.sub_structures[l-1][0]._magnetization["amplitude"]==0:
                        M2[:,:,:,l]=0
                    if S.sub_structures[l-1][0]._magnetization["amplitude"]!=0:
                        M2[:,:,:,l]=M_temp[:,:,:,(l-n_caps-1)//l2]
                    temp=S.sub_structures[l-1][0]._atom.get_atomic_form_factor(self.energy)
                    if np.imag(temp)<0:
                        temp=np.conj(temp)
                    f_Charge[:,:,l]=temp
                    
                    temp2=S.sub_structures[l-1][0]._atom.get_magnetic_form_factor(self.energy)
                    
                    if temp2!=0 and "f_manual_input" in params_general and params_general["f_manual_input"]==True:
                        f_Mag[:,:,l]=params_general["f_mag_manual"]
                        f_Charge[:,:,l]=params_general["f_charge_manual"]
                    if "f_manual_input" not in params_general or params_general["f_manual_input"]==False:
                        f_Mag[:,:,l]=temp2
                    if f_Mag[:,:,l].sum()!=0:
                        f_Charge_maglayer_scalar=f_Charge[0,0,l]
                        f_Mag_scalar=f_Mag[0,0,l]
                        na_scalar=S.sub_structures[l-1][0]._density/S.sub_structures[l-1][0].atom.mass.magnitude
                    f_Mag2[:,:,l]=complex(0,0)
                    na[:,:,l]=S.sub_structures[l-1][0]._density/S.sub_structures[l-1][0].atom.mass.magnitude
                    
            self.dz_mag=params_3D["dz_mag"]*1e-9   
            self.dz=params_3D["dz_average"]*1e-9      
            M3=M2.copy()
            M3[0,:,:,:]=M2[0,:,:,:]*np.cos(self.phi_rotate*np.pi/180)+M2[1,:,:,:]*np.sin(self.phi_rotate*np.pi/180)
            M3[1,:,:,:]=-M2[0,:,:,:]*np.sin(self.phi_rotate*np.pi/180)+M2[1,:,:,:]*np.cos(self.phi_rotate*np.pi/180)
            self.M=(M3)
            self.na=na
            self.f_Charge=f_Charge
            self.f_Mag=f_Mag
            self.f_Mag2=f_Mag2
            self.f_Charge_maglayer_scalar=f_Charge_maglayer_scalar
            self.f_Mag_scalar=f_Mag_scalar  
            self.na_scalar=na_scalar
def Sample2Reflection_Coefficients(simulation_input,outputs=["R","output1","output2","bkg1","bkg2"]):
#This method runs as a Main program to calculate the XRMS signal for a given set of simulation parameters and 
#sample, all bundled together in a single object defined in a simulation definition script, with the "sample" attribute
#an instance of the "Generic_Sample" class
    R=[]#initializing the list into which the reflection coefficents will be exported
    output1=[]
    output2=[]
    bkg1=[]
    bkg2=[]
    if simulation_input['Simulation_Parameters'].get("phi_rotations")==None:
        simulation_input['Simulation_Parameters']["phi_rotations"]=[0]
    for count3,phi_deg in enumerate(simulation_input['Simulation_Parameters']["phi_rotations"]):
        simulation_input['Simulation_Parameters']["phi_rotate"]=phi_deg
        sample=Generic_sample(simulation_input)
        for count2,theta_deg in enumerate(simulation_input['Simulation_Parameters']["angles"]):
            print(theta_deg)
            theta=theta_deg*np.pi/180
            energy_simulation=simulation_input['Simulation_Parameters']["energy"]
            size=simulation_input['3D_Sample_Parameters']["shape"]
            XRMS=XRMS_Simulate(sample,theta,energy=energy_simulation,full_column=sample.full_column)
            #initializing simulation class
            
            XRMS.Chi_define()
            XRMS.get_A_S_matrix()
            XRMS.XM_define()
            XRMS.Matrix_stack(stack_coordinates=sample.stack_coordinates)
            if sample.full_column=="full":
                XRMS.get_R()
                if simulation_input['Simulation_Parameters']["rotated_stripes"]==True:    
                    if simulation_input['Simulation_Parameters']["differential_absorption"]==True:
                    
                        XRMS.get_Faraday_rotated_stripe()
                    temp1,temp2=XRMS.Ewald_sphere_rotated_stripe(XRMS.R_array[0,::],sample, simulation_input)
                        
                else:
                    if simulation_input['Simulation_Parameters']["differential_absorption"]==True:
                        XRMS.get_Faraday_Parallel()
                    
                    if "R" in outputs:
                        R.append(XRMS.R_array)
                        
                        #INSERT R expansion and rotation here
                        # this method must also update basic simulation parameters from the original to the expanded ones
                        
                    temp1,temp2=XRMS.Ewald_sphere_pixel_index(XRMS.R_array,sample, simulation_input)
            if "output1" in outputs:
                output1.append(temp1)
            if "output2" in outputs:
                output2.append(temp2)
            if simulation_input['Simulation_Parameters']["calculate_background"]==True:
                bk1,bk2=background_Lee.background_main(simulation_input,theta=theta_deg*np.pi/180)
                if "bkg1" in outputs:
                    bkg1.append(bk1)
                if "bkg2" in outputs:
                    bkg2.append(bk2)
            else:
                det_size=simulation_input['Simulation_Parameters']["det_size"]
                bk1=np.zeros((det_size[0],det_size[1]))
                bk2=np.zeros((det_size[0],det_size[1]))
                if "bkg1" in outputs:
                    bkg1.append(bk1)
                if "bkg2" in outputs:
                    bkg2.append(bk2)
                
    return R,output1,output2,bkg1,bkg2


             
        
        