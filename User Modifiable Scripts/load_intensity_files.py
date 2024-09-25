import numpy as np
import matplotlib.pyplot as plt
import csv

I_files1=["../Magnetization Files/Intensity10.csv","../Magnetization Files/Intensity11.csv"]
I_files2=["../Magnetization Files/Intensity20.csv","../Magnetization Files/Intensity21.csv"]
bkgfile1=["../Magnetization Files/IB10.csv","../Magnetization Files/IB11.csv"]
bkgfile2=["../Magnetization Files/IB20.csv","../Magnetization Files/IB21.csv"]
for k in range(2):
    beam_stop=np.ones((256,256))
    beam_stop[128-100:128+100,128-16:128+16]=0
    
    with open(I_files1[k]) as f:
        reader = csv.reader(f)
        data = list(reader)
        data=np.array(data,dtype=float)
        data_reshaped=data.reshape((256,256),order="C")
    Intensity1=data_reshaped
    Intensity1[128,128]=Intensity1[128,128]/100
    with open(I_files2[k]) as f:
        reader = csv.reader(f)
        data = list(reader)
        data=np.array(data,dtype=float)
        data_reshaped=data.reshape((256,256),order="C")
    Intensity2=data_reshaped
    Intensity2[128,128]=Intensity2[128,128]/100
    with open(bkgfile1[k]) as f:
        reader = csv.reader(f)
        data = list(reader)
        data=np.array(data,dtype=float)
        data_reshaped=data.reshape((254,254),order="C")
    background1=data_reshaped/50
    with open(bkgfile2[k]) as f:
        reader = csv.reader(f)
        data = list(reader)
        data=np.array(data,dtype=float)
        data_reshaped=data.reshape((254,254),order="C")
    background2=data_reshaped/50
    
    #calculating the outgoing intensities combining the results for the polarizations
    IB1=np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(Intensity1[1:-1,1:-1])*np.fft.fft2(background1)))
    IB2=np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(Intensity2[1:-1,1:-1])*np.fft.fft2(background2)))
    Intensities_Sum_bkg=np.real(IB1+IB2)
    Intensities_Diff_bkg=np.real(IB1-IB2)

    Intensities_Sum=np.real(Intensity1+Intensity2)
    Intensities_Diff=np.real(Intensity1-Intensity2)

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,10))
    extent=[-128,128,-128,128]
    im=ax1.imshow((np.abs(Intensities_Sum*beam_stop)),extent=extent)
    xticks=list(np.linspace(-128,128,15))
    xticklabels=[]
    for j in range(15):
        xticklabels.append(str(j-7))
    yticks=list(np.linspace(-128,128,15))
    yticklabels=[]
    for j in range(15):
        yticklabels.append(str(j-7))
    ax1.set_title("Sum of both circular polarizations \n (lin colourscale)",fontsize=12)
    ax1.set_xlabel("Detector_x (mm) (20cm sample_det distance)",fontsize=10)
    ax1.set_ylabel("Detector_y (mm) (20cm sample_det distance)",fontsize=10)
    ax1.set(xticks=xticks,yticks=yticks)
    ax1.set_xticklabels(xticklabels,fontsize=8)
    ax1.set_yticklabels(yticklabels,fontsize=8)

    im=ax2.imshow(np.abs(Intensities_Diff*beam_stop)**1.0*np.sign(Intensities_Diff),extent=extent)
    ax2.set_title("Difference of both circular polarizations \n (lin colourscale)",fontsize=12)
    ax2.set_xlabel("Detector_x (mm)  (20cm sample_det distance)",fontsize=10)
    ax2.set_ylabel("Detector_y (mm)  (20cm sample_det distance)",fontsize=10)
    ax2.set(xticks=xticks,yticks=yticks)
    ax2.set_xticklabels(xticklabels,fontsize=8)
    ax2.set_yticklabels(yticklabels,fontsize=8)

    filename="../figures/worms_run4"+str(k)
    plt.savefig(filename+"sumdiff.png")

    im=ax1.imshow((np.abs(Intensities_Sum_bkg*beam_stop[1:-1,1:-1])),extent=extent)
    ax1.set_title("Sum of both circular polarizations \n (lin colourscale)",fontsize=12)
    ax1.set_xlabel("Detector_x (mm) (20cm sample_det distance)",fontsize=10)
    ax1.set_ylabel("Detector_y (mm) (20cm sample_det distance)",fontsize=10)
    ax1.set(xticks=xticks,yticks=yticks)
    ax1.set_xticklabels(xticklabels,fontsize=8)
    ax1.set_yticklabels(yticklabels,fontsize=8)

    im=ax2.imshow(np.abs(Intensities_Diff_bkg*beam_stop[1:-1,1:-1])**1.0*np.sign(Intensities_Diff_bkg),extent=extent)
    ax2.set_title("Difference of both circular polarizations \n (lin colourscale)",fontsize=12)
    ax2.set_xlabel("Detector_x (mm)  (20cm sample_det distance)",fontsize=10)
    ax2.set_ylabel("Detector_y (mm)  (20cm sample_det distance)",fontsize=10)
    ax2.set(xticks=xticks,yticks=yticks)
    ax2.set_xticklabels(xticklabels,fontsize=8)
    ax2.set_yticklabels(yticklabels,fontsize=8)

    filename="../figures/worms_run4_bkg"+str(k)
    plt.savefig(filename+"sumdiff.png")