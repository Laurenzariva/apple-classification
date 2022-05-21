# Perbandingan klasifikasi Naive-Bayes dan KNN untuk Mengidentifikasi Jenis Buah Apel 
# dengan ekstraksi ciri LBP dan HSV

import numpy as np
import cv2 as cv
import pandas as pd
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import seaborn as sns
import statistics

fuji_histLBP = []
manalagi_histLBP = []
RedDelicious_histLBP = []
RomeBeauty_histLBP = []
GrannySmith_histLBP = []

fuji_histHSV = []
manalagi_histHSV = []
RedDelicious_histHSV = []
RomeBeauty_histHSV = []
GrannySmith_histHSV = []

# settings for LBP
radius = 3
n_points = 8 * radius
METHOD = 'uniform'

for apel in range(20):
    fuji_image = "Dataset_Apel/Fuji/fuji%i.jpg" %(apel+1)
    fuji_LBP=local_binary_pattern(cv.imread(fuji_image, cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
    fuji_HSV = cv.cvtColor(cv.imread(fuji_image), cv.COLOR_BGR2HSV)
    histLBP_fuji, bins = np.histogram(fuji_LBP.ravel(),256,[0,256])
    histHSV_fuji, bins_ = np.histogram(fuji_HSV.ravel(),256,[0,256])
    fuji_histLBP.append(np.transpose(histLBP_fuji[0:18,np.newaxis]))
    fuji_histHSV.append(np.transpose(histHSV_fuji[0:18,np.newaxis]))
    
    manalagi_image = "Dataset_Apel/Manalagi/manalagi%i.jpg" %(apel+1)
    manalagi_LBP=local_binary_pattern(cv.imread(manalagi_image, cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
    manalagi_HSV = cv.cvtColor(cv.imread(manalagi_image), cv.COLOR_BGR2HSV)
    histLBP_manalagi, bins = np.histogram(manalagi_LBP.ravel(),256,[0,256])
    histHSV_manalagi, bins_ = np.histogram(manalagi_HSV.ravel(),256,[0,256])
    manalagi_histLBP.append(np.transpose(histLBP_manalagi[0:18,np.newaxis]))
    manalagi_histHSV.append(np.transpose(histHSV_manalagi[0:18,np.newaxis]))
    
    RedDelicious_image = "Dataset_Apel/Red_Delicious/RedDelicious%i.jpg" %(apel+1)
    RedDelicious_LBP = local_binary_pattern(cv.imread(RedDelicious_image, cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
    RedDelicious_HSV = cv.cvtColor(cv.imread(RedDelicious_image), cv.COLOR_BGR2HSV)
    histLBP_RedDelicious, bins = np.histogram(RedDelicious_LBP.ravel(),256,[0,256])
    histHSV_RedDelicious, bins_ = np.histogram(RedDelicious_HSV.ravel(),256,[0,256])
    RedDelicious_histLBP.append(np.transpose(histLBP_RedDelicious[0:18,np.newaxis]))
    RedDelicious_histHSV.append(np.transpose(histHSV_RedDelicious[0:18,np.newaxis]))
    
    RomeBeauty_image = "Dataset_Apel/Rome_Beauty/RomeBeauty%i.jpg" %(apel+1)
    RomeBeauty_LBP = local_binary_pattern(cv.imread(RomeBeauty_image, cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
    RomeBeauty_HSV = cv.cvtColor(cv.imread(RomeBeauty_image), cv.COLOR_BGR2HSV)
    histLBP_RomeBeauty, bins = np.histogram(RomeBeauty_LBP.ravel(),256,[0,256])
    histHSV_RomeBeauty, bins_ = np.histogram(RomeBeauty_HSV.ravel(),256,[0,256])
    RomeBeauty_histLBP.append(np.transpose(histLBP_RomeBeauty[0:18,np.newaxis]))
    RomeBeauty_histHSV.append(np.transpose(histHSV_RomeBeauty[0:18,np.newaxis]))
    
    GrannySmith_image = "Dataset_Apel/Granny_Smith/GrannySmith%i.jpg" %(apel+1)
    GrannySmith_LBP = local_binary_pattern(cv.imread(GrannySmith_image, cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
    GrannySmith_HSV = cv.cvtColor(cv.imread(GrannySmith_image), cv.COLOR_BGR2HSV)
    histLBP_GrannySmith, bins = np.histogram(GrannySmith_LBP.ravel(),256,[0,256])
    histHSV_GrannySmith, bins_ = np.histogram(GrannySmith_HSV.ravel(),256,[0,256])
    GrannySmith_histLBP.append(np.transpose(histLBP_GrannySmith[0:18,np.newaxis]))
    GrannySmith_histHSV.append(np.transpose(histHSV_GrannySmith[0:18,np.newaxis]))

DataTrain_lbp = np.concatenate((fuji_histLBP + manalagi_histLBP + RedDelicious_histLBP + RomeBeauty_histLBP + GrannySmith_histLBP), axis=0).astype(np.float32)
DataTrain_hsv = np.concatenate((fuji_histHSV + manalagi_histHSV + RedDelicious_histHSV + RomeBeauty_histHSV + GrannySmith_histHSV), axis=0).astype(np.float32)

kelasTrain = []
for fuji in range(20):
    kelasTrain.append("Fuji")
for manalagi in range(20):
    kelasTrain.append("Manalagi")
for RedDelicious in range(20):
    kelasTrain.append("Red Delicious")
for RomeBeauty in range(20):
    kelasTrain.append("Rome Beauty")
for GrannySmith in range(20):
    kelasTrain.append("Granny Smith")
responsesTrain = np.array(kelasTrain)

# Untuk menampilkan histogram
# plt.hist(DataTrain_lbp.ravel(),bins=256)
# plt.show()
# plt.hist(DataTrain_hsv.ravel(), bins=256)
# plt.show()

print("\nEvaluasi : \n")

def cari_kOptimum(kf, DataTrain):           # Mencari k optimum dari data train baik LBP maupun HSV
    nilai_k = np.zeros(10)
    for i,(train_index, test_index) in enumerate(kf.split(DataTrain)):
        Xtrain, Xtest = DataTrain[train_index], DataTrain[test_index]
        ytrain, ytest = responsesTrain[train_index], responsesTrain[test_index]
        
        for k in range (10):
            knnLBP = KNeighborsClassifier(n_neighbors=(k+1))
            knnLBP.fit(Xtrain,ytrain)
            yp = knnLBP.predict(Xtest)
            acc = accuracy_score(ytest,yp)
            nilai_k[k] = nilai_k[k] + acc
            
    mean_nilai_k=[]
    for k in range(10):
        mean_nilai_k.append(nilai_k[k]/nFold)

    maksKNN = mean_nilai_k.index(max(mean_nilai_k[1:10]))+1
    return maksKNN


nFold = 5
kf = KFold(n_splits=nFold, shuffle=True,random_state=0)

print("\n -- KNN LBP --- \n")       # Melihat akurasi dan matriks confusion KNN dengan LBP
scores = np.zeros(5)
plt.figure(figsize=(12,2))

for i,(train_index, test_index) in enumerate(kf.split(DataTrain_lbp)):
    Xtrain, Xtest = DataTrain_lbp[train_index], DataTrain_lbp[test_index]
    ytrain, ytest = responsesTrain[train_index], responsesTrain[test_index]
    
    knnLBP = KNeighborsClassifier(n_neighbors=(cari_kOptimum(kf, DataTrain_lbp)))
    knnLBP.fit(Xtrain,ytrain)
    yp = knnLBP.predict(Xtest)
    acc = accuracy_score(ytest,yp)
    scores[i] = acc
    
    plt.subplot(1,5,i+1)
    cm = confusion_matrix(yp,ytest) 
    sns.heatmap(cm,annot=True)        

plt.show()
print(f'Akurasi KNN LBP --> {scores}')
print('Rata-rata akurasi KNN LBP: %.2f%%' %(np.mean(scores*100)))
print()


print("\n -- KNN HSV -- \n")        # Melihat akurasi dan matriks confusion KNN dengan HSV
scores = np.zeros(5)
plt.figure(figsize=(12,2))

for i,(train_index, test_index) in enumerate(kf.split(DataTrain_hsv)):
    Xtrain, Xtest = DataTrain_hsv[train_index], DataTrain_hsv[test_index]
    ytrain, ytest = responsesTrain[train_index], responsesTrain[test_index]
    
    knnHSV = KNeighborsClassifier(n_neighbors=(cari_kOptimum(kf, DataTrain_hsv)))
    knnHSV.fit(Xtrain,ytrain)
    yp = knnHSV.predict(Xtest)
    acc = accuracy_score(ytest,yp)
    scores[i] = acc
    
    plt.subplot(1,5,i+1)
    cm = confusion_matrix(yp,ytest) 
    sns.heatmap(cm,annot=True)        

plt.show()
print(f'Akurasi KNN HSV --> {scores}')
print('Rata-rata akurasi KNN HSV: %.2f%%' %(np.mean(scores*100)))
print()


print("\n -- NB LBP -- \n")         # Melihat akurasi dan matriks confusion Naive Bayes dengan LBP
nbLBP = GaussianNB()
scores = np.zeros(5)
plt.figure(figsize=(12,2))
for i,(train_index, test_index) in enumerate(kf.split(DataTrain_lbp)):
    Xtrain, Xtest = DataTrain_lbp[train_index], DataTrain_lbp[test_index]
    ytrain, ytest = responsesTrain[train_index], responsesTrain[test_index]
    
    nbLBP.fit(Xtrain,ytrain)
    yp = nbLBP.predict(Xtest)
    acc = accuracy_score(ytest,yp)
    scores[i] = acc

    plt.subplot(1,5,i+1)
    cm = confusion_matrix(yp,ytest) 
    sns.heatmap(cm,annot=True)

plt.show()
print('Akurasi NB LBP --> ', scores)
print('Rata-rata akurasi NB LBP : %.2f%%' %(np.mean(scores*100)))
print()


print("\n -- NB HSV -- \n")     # Melihat akurasi dan matriks confusion Naive Bayes dengan HSV
nbHSV = GaussianNB()
scores = np.zeros(5)
plt.figure(figsize=(12,2))
for i,(train_index, test_index) in enumerate(kf.split(DataTrain_hsv)):
    Xtrain, Xtest = DataTrain_hsv[train_index], DataTrain_hsv[test_index]
    ytrain, ytest = responsesTrain[train_index], responsesTrain[test_index]
    
    nbHSV.fit(Xtrain,ytrain)
    yp = nbHSV.predict(Xtest)
    acc = accuracy_score(ytest,yp)
    scores[i] = acc

    plt.subplot(1,5,i+1)
    cm = confusion_matrix(yp,ytest) 
    sns.heatmap(cm,annot=True)

plt.show()
print('Akurasi NB HSV --> ', scores)
print('Rata-rata akurasi NB HSV : %.2f%%' %(np.mean(scores*100)))
print()

