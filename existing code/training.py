import cv2
#globbing utility.
import glob
import csv 
from skimage.feature import greycomatrix, greycoprops
import numpy as np

from skimage import color
from skimage import io
import numpy as np 

import pandas as pd
import cv2 as cv

contrasts=[]
Dissimilaritys=[]
Homogeneitys=[]
ASMs=[]
Energys=[]
Correlations=[]
proList = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy']
featlist = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy','hue','value', 'saturaton','path','label']
properties =np.zeros(6)
glcmMatrix = []
final=[]
#abc = io.imread('14.jpg')


'''

j=[]
for j in range(j, len(proList)):
            properties[j] = (greycoprops(glcmMatrix, prop=proList[j]))

            features = np.array([properties[0], properties[1], properties[2], properties[3], properties[4],h_mean,s_mean,v_mean])
            final.append(features)


        # images = images.f.arr_0
        print(image_folder_list[i])


        glcmMatrix = (greycomatrix(gray_image, [1], [0], levels=256))
        
'''
path="*.jpg"

    


finalcorrelation=[];   
for file in glob.glob(path):
    #img = color.rgb2gray(io.imread(file))
    im = cv2.imread(file)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ngcm= greycomatrix(im, [1], [0], 256, symmetric=False, normed=True)
    result=ngcm
    
    contrast = greycoprops(result, 'contrast')
    contrast=np.mean(contrast)
    contrasts.append(contrast)
    Dissimilarity=greycoprops(result, 'dissimilarity')
    Dissimilarity=np.mean(Dissimilarity)
    Dissimilaritys.append(Dissimilarity)
    Homogeneity=greycoprops(result,'homogeneity')
    Homogeneity=np.mean(Homogeneity)
    Homogeneitys.append(Homogeneity)
    ASM=greycoprops(result,'ASM')
    ASM=np.mean(ASM)
    ASMs.append(ASM)
    Energy=greycoprops(result,'energy')
    Energy=np.mean(Energy)
    Energys.append(Energy)
    Correlation=greycoprops(result,'correlation')
    Correlation=np.mean(Correlation)
    Correlations.append(Correlation)
    finalcontrasts = pd.DataFrame(contrasts)
    finalDissimilaritys = pd.DataFrame(Dissimilaritys)
    finalHomogeneitys = pd.DataFrame(Homogeneitys)
    finalASMs = pd.DataFrame(ASMs)
    finalEnergy=pd.DataFrame(Energys)
    #finalCorrelations = pd.DataFrame(Correlations)
    
r = []
N = 24
import random 
def twoRandomNumbers(a,b): 
    test = random.random() # random float 0.0 <= x < 1.0 
    
    if test < 0.5: 
        return a 
    else: 
        return b
for x in range(N):
    a=twoRandomNumbers(0, 1)
    #print(a)
    r.append(a)

finaltarget = pd.DataFrame(r)
#Data = [["Contrast", "Dissimilarity","Homogeneity","ASM","Target"],[finalcontrasts, finalDissimilaritys, finalHomogeneitys,finalASMs,finaltarget]]


#frames = [finalcontrasts, finalDissimilaritys, finalHomogeneitys,finalASMs,finalCorrelations,finaltarget,finaltarget]
frames = [finalcontrasts, finalDissimilaritys, finalHomogeneitys,finalASMs,finalEnergy,finaltarget]
finalresult = pd.concat(frames, axis=1)    
finalresult.to_csv('testing.csv', mode='a', header=False)


    