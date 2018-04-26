import numpy as np
import pandas as pd
import os
import glob
import cv2
np.set_printoptions(threshold=np.nan)

import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

import dicom
dicom_root = 'test/'
patients = [ x for x in os.listdir(dicom_root)]
[print(x) for x in os.listdir(dicom_root)]
print('Patient count: {}'.format(len(patients)))

def rescale_correction(s):
    s.image = s.pixel_array * 1 - 1024

def load_patient(patient_id):
    files = glob.glob(dicom_root + '/{}/*.dcm'.format(patient_id))
    slices = []
    for f in files:
        dcm = dicom.read_file(f)
        rescale_correction(dcm)
        slices.append(dcm)
    
    slices = sorted(slices, key=lambda x: x.SliceLocation)
    return slices

for patient_no in patients:
    pat = load_patient(patient_no)
    print(patient_no)
    print(len(pat)/2)

    for i in range(0,22):

        img = pat[i].image.copy()
        img[img<0]=0
        
        Z = np.reshape(img,(1,262144))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        for j in range(2,8):
            print(j)
            ret,label,center=cv2.kmeans(Z,j,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

         
            center = np.uint8(center)
            res = center[label.flatten()]
            res2 = res.reshape((img.shape))
            cv2.imwrite('res2.jpg',res2)            
            img=res2
            img = np.uint8(img)
           
            im2, contours, _ = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros(img.shape, np.uint8)
            cv2.fillPoly(mask, contours, 255)
            
            img = ~img
            img[(mask == 0)] = 0 

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  
            img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
            img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
            img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
            
            img2 = pat[i].image.copy()
            img2[(img == 0)] = 0 

            plt.figure(figsize=(12, 12))
            plt.subplot(131)
            plt.imshow(pat[i].image)
            print('orginal'+str(i)+'.jpg')
            cv2.imwrite('orginal'+str(i)+'.jpg',pat[i].image)
            plt.subplot(132)
            plt.imshow(img)
            print('mask'+str(i)+' '+str(j)+'.jpg')
            cv2.imwrite('mask'+str(i)+' '+str(j)+'.jpg',img)
            plt.subplot(133)
            plt.imshow(img2)
            print('tumor'+str(i)+' '+str(j)+'.jpg')
            cv2.imwrite('tumor'+str(i)+' '+str(j)+'.jpg',img2)
            
