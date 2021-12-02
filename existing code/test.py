
import cv2
#globbing utility.
import glob
import csv 
from skimage.feature import greycomatrix, greycoprops
import numpy as np
from sklearn import svm
from skimage import color
from skimage import io
import numpy as np 
from skimage.filters import threshold_otsu
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
#im = cv2.imread('mdb321.jpg')
im = io.imread('1.jpg')
plt.imshow(im)
plt.title("Input Image") 
plt.show()
Gaussian = cv2.GaussianBlur(im,(5,5),0)
plt.imshow(Gaussian)
plt.title("Pre-processed Image") 
plt.show()
# Reshaping the image into a 2D array of pixels and 3 color values (RGB) 
pixel_vals = Gaussian.reshape((-1,3))  
# Convert to float type 
pixel_vals = np.float32(pixel_vals)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)  
# then perform k-means clustering wit h number of clusters defined as 3 
#also random centres are initally chosed for k-means clustering 
k = 3
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)   
# convert data into 8-bit values 
centers = np.uint8(centers) 
segmented_data = centers[labels.flatten()]   
# reshape data into the original image dimensions 
segmented_image = segmented_data.reshape((Gaussian.shape)) 
plt.imshow(segmented_image)
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ngcm= greycomatrix(im, [1], [0], 256, symmetric=False, normed=True)
result=ngcm
contrast = greycoprops(result, 'contrast')
contrast=np.mean(contrast)
Dissimilarity=greycoprops(result, 'dissimilarity')
Dissimilarity=np.mean(Dissimilarity)
Homogeneity=greycoprops(result,'homogeneity')
Homogeneity=np.mean(Homogeneity)
ASM=greycoprops(result,'ASM')
ASM=np.mean(ASM)
Energy=greycoprops(result,'energy')
Energy=np.mean(Energy)
Correlation=greycoprops(result,'correlation')
Correlation=np.mean(Correlation)
#finalvalue=[contrast, Dissimilarity,Homogeneity,ASM,Energy,Correlation]
finalvalue=[contrast, Dissimilarity,Homogeneity,ASM,Energy]

df=pd.read_csv('testing.csv',index_col=0)
X, y = df.iloc[:, :-1], df.iloc[:, -1]
print(X.shape)
print(y.shape)

clf = svm.SVC(kernel='linear') # Linear Kernel
abc=clf.fit(X, y)
abc1=abc.predict([finalvalue])
print(abc1)

if abc1 == 0:
  print("Input image is Not affected")
elif abc1 == 1:
  print("Input image is affected")
 
df=pd.read_csv('testing.csv',index_col=0)
X, y = df.iloc[:, :-1], df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=42)
print ("train_set_x shape: " + str(X_train.shape))
print ("train_set_y shape: " + str(y_train.shape))
print ("test_set_x shape: " + str(X_test.shape))
print ("test_set_y shape: " + str(y_test.shape))
accuracy = []
# list of algorithms names
classifiers = ['Voting']
# loop through algorithms and append the score into the list model.fit(X_train, y_train)
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
eclf3 = KNeighborsClassifier(n_neighbors=3)
eclf3 = eclf3.fit(X_train, y_train)
predicted =eclf3.predict(X_test)
print("KNN Classifier accuracy is")
print(accuracy_score(y_test,predicted)*100)
print(classification_report(y_test,predicted))
print(confusion_matrix(y_test,predicted))
