# 3D_XRMS
Python package for 3D X-ray resonant magnetic scattering simulations

The aim with this project is to create a publishable package for simulating 3D XRMS measurements. 

The proposed structure is as follows:
1. A "sample" class where the user inputs all details of their sample. In the future, the ideal would be to design a GUI to generate this class
2. A class for the simulation of the specular reflection using the Stepanov Sinha algorithm. This is a prerequisite for simulating the full 3D reflection
3. The main 3D scattering simulation class
4. The background simulation package based upon the Lee/Stepanov/Sinha model

