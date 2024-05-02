# 3D_XRMS
Python package for 3D X-ray resonant magnetic scattering simulations

This code is based upon the following 3 publications: 

1. https://journals.aps.org/prb/abstract/10.1103/PhysRevB.61.15302
2. https://journals.aps.org/prb/abstract/10.1103/PhysRevB.68.224410
3. https://journals.aps.org/prb/cited-by/10.1103/PhysRevB.103.184401

Starting from a 3D micromagnetic simulation, or alternatively an analytically defined magneitzation distribution as generated for example by the the script "analytic domain pattern.py", users must then save the magnetization distribution as a set of 3 .csv files Mx, My and Mz, where the 3D array of magnetization is put in 1D order via "c language" ordering. 

For running the simulation package, a series of scripts have been produced which users can use as templates. Each of these has been extensively commented.

Multilayer_simple.py is set up to run on the output of analytic domain pattern.py, where the sample should approximate the experimental sample used in reference 3. 
Sample input script.py is designed to run on the output of micromagnetic simulations saved in the 3 files [Mx_LSMO.csv, My_LSMO.csv, Mz_LSMO.csv], using the real crystal lattice as the multilayer
Sample input script multilayer.py is designed to run on the output of micromagnetic simulations saved in the 3 files Mx_LSMO.csv, My_LSMO.csv, Mz_LSMO.csv, however using a user defined multlayer as the reflection object
complex sample input script multilayer.py runs on the micromagnetic simulations saved in [Mx_Matlab.csv, My_Matlab.csv, Mz_Matlab.csv].

In these input scripts, there are a range of adjustable parameters such as the incident angles, the energy, whether the Henke files for the scattering factors are to be used or rather user defined values etc. Users must also decide whether they wish to calculate the background scatter (sometimes of substantial importance for multilayers - particularly when an irregular domain pattern is present), and whether they wish to account for differential absorption effects which can be important in cases where an applied field is present.    

