# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 10:48:39 2023

@author: sflewett
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

def get_filenames(directory,file_index):
    #this function gets the full filenames 
    #from those stored in the relevant directory
     path = directory
     file_name_list=[]
     for file_names in os.listdir(path):
         file_name_list.append(file_names)
       
     return file_name_list
 
    
def file_list_generate(header,footer,filenumbers):
    #this is specific to this particular file format from 
    #the SEXTANTS beamline
    #filenumbers is a list or numpy array containing the filenumbers
    #the header and footer are read off from the filenames obtained by
    #get_filenames
    
        
    filenames=[]
    filestrings=[]
    for filenumber in filenumbers:
        if filenumber <10:
            filestring="000"+str(filenumber)
        elif filenumber<100:
            filestring="00"+str(filenumber)
        elif filenumber < 1000:
            filestring="0"+str(filenumber)
        else:
            filestring=str(filenumber)
        #in order to convert from a number to a string, we need to add zeros
        filenames.append(header+filestring+footer)
        filestrings.append(filestring)
    return filenames,filestrings

def data_extract(path,filenames,filestrings):
    #given a filename, we extract the relevant data from the hdf file
    count=0
    depth=len(filenames)
    diffraction_patterns=[]
    for file in range(depth):
        print(filenames[file])
        f = h5py.File(path+filenames[file], 'r')
        diffraction=(f["scan_"+filestrings[count]]["scan_data"]["data_06"])[0,:,:]
        #opening and extracting the diffraction data
        #the exact keys are specific to SEXTANTS
        diffraction=np.array(diffraction)
        diffraction2 = diffraction.astype('f')
        #converting from integer to floating point
        diffraction_patterns.append(diffraction2)
        f.close()
        count=count+1
        #print(count)
    return np.array(diffraction_patterns)
    #we return a 3D array of the measured diffraction data
def visitor_func(name, node):
    #function to list elements of the hdf file
    if isinstance(node, h5py.Group):
        print(node.name, 'is a Group')
    elif isinstance(node, h5py.Dataset):
       if (node.dtype == 'object') :
            print (node.name, 'is an object Dataset')
       else:
            print(node.name, 'is a Dataset')   
    else:
        print(node.name, 'is an unknown type')         

#####    



path="D:/RESOXS_20230620_LSMO/"

header="scanx_"
footer=".nxs"
#we know to use these headers and footers because we used "get_filenames" 
filenumbers=list(range(1391,1871))#start again at 2535 (2534 corrupted)
#these numbers will change depending upon the data you want to look at
filenames,filestrings=file_list_generate(header,footer,filenumbers)
data_array=data_extract(path,filenames,filestrings)
#data_array is a 3D numpy array containing all of the raw diffraction data

delta=1
data_array1=data_array[::4,:,:]
data_array2=data_array[1::4,:,:]
data_array3=data_array[2::4,:,:]
data_array4=data_array[3::4,:,:]
#splitting the array in 4 parts for energy and helicities
#please see the macros in the logbook for the justification

ass1=np.zeros(data_array1.shape)
ass2=np.zeros(data_array1.shape)
#initializing the asymmetry ratio arrays

for j in range(data_array1.shape[0]):
    app_background1=np.sum(data_array1[j,0:200,1700:1900])/200.0**2
    app_background2=np.sum(data_array2[j,0:200,1700:1900])/200.0**2
    app_background3=np.sum(data_array3[j,0:200,1700:1900])/200.0**2
    app_background4=np.sum(data_array4[j,0:200,1700:1900])/200.0**2
    #very rudimentary background subtraction. This will need to be done properly
    #in order to get reliable values of the asymmetry ratios
    
    ass2[j,:,:]=(data_array4[j,:,:]-data_array3[j,:,:])/((data_array4[j,:,:]+data_array3[j,:,:]-app_background3-app_background4)+delta)
    ass1[j,:,:]=(data_array2[j,:,:]-data_array1[j,:,:])/((data_array2[j,:,:]+data_array1[j,:,:]-app_background1-app_background2)+delta)


ass1[np.abs(ass1)>1.0]=0.0
ass2[np.abs(ass2)>1.0]=0.0 
#data_array1[:,:,820:1040]=0 
#data_array2[:,:,820:1040]=0 
#data_array3[:,:,820:1040]=0 
#data_array4[:,:,820:1040]=0 
ass1[:,0,0]=0.7
ass1[:,0,0]=-0.7
#normalizing the colour scale

side_by_side=ass1-ass1
side_by_side=data_array1-data_array2
#side_by_side[:,:,300:600]=data_array3[:,:,1000:1300]+data_array4[:,:,1000:1300]
#setting up the asymmetry ratios to be plotted side by side


del data_array
data_array1[:,700:1300,850:1050]=0
data_array2[:,700:1300,850:1050]=0
data_array3[:,700:1300,850:1050]=0
data_array4[:,700:1300,850:1050]=0

#here I was playing with using a digital beamstop to block out the central part of the beam

side_by_side[:,:,800:1050]=0
for j in range(side_by_side.shape[0]):
    #plt.matshow(data_array4[j,:,:]-data_array3[j,:,:])
    #plt.matshow(data_array2[j,:,:]-data_array1[j,:,:])
    plt.matshow(side_by_side[j,:,:])
    #plt.close()
    #displaying the results
    


