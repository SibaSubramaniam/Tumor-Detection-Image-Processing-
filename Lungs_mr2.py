# Standard imports
import numpy as np
import pandas as pd
import os
import glob
import cv2
np.set_printoptions(threshold=np.nan)

# Imaging libraries
import matplotlib.pyplot as plt
#matplotlib inline
#import seaborn as sns
#p = sns.color_palette()
#import plotly.offline as py
#import plotly.graph_objs as go
#py.init_notebook_mode(connected=True)

# Pandas configuration
pd.set_option('display.max_columns', None)
#print('OK.')
#print(cv2.__version__)

# get patients list
import dicom
dicom_root = 'test/'
patients = [ x for x in os.listdir(dicom_root)]
[print(x) for x in os.listdir(dicom_root)]
print('Patient count: {}'.format(len(patients)))


# DICOM rescale correction
def rescale_correction(s):
    s.image = s.pixel_array * 1 - 1024

# Returns a list of images for that patient_id, in ascending order of Slice Location
# The pre-processed images are stored in ".image" attribute
def load_patient(patient_id):
    files = glob.glob(dicom_root + '/{}/*.dcm'.format(patient_id))
    slices = []
    for f in files:
        dcm = dicom.read_file(f)
        #print(dcm)
        rescale_correction(dcm)
        # TODO: spacing eq.
        slices.append(dcm)
    
    slices = sorted(slices, key=lambda x: x.SliceLocation)
    return slices


# Load a patient
for patient_no in patients:
    pat = load_patient(patient_no)
    print(patient_no)
    print(len(pat)/2)

    img = pat[int(len(pat)/2)].image.copy()

    print(img)
    #cv2.imwrite('47'+'.jpg',img)
    #print(img.filename)

    # threshold HU > -300
    '''
    img[img==20] = 150
    img[img==21] = 150
    img[img==22] = 150
    img[img==23] = 150
    img[img==24] = 150
    img[img==25] = 150
    img[img==26] = 150
    img[img==27] = 150 
    img[img==28] = 150
    img[img==29] = 150
    img[img==30] = 150
    
    img[img==31] = 200
    img[img==32] = 200
    img[img==33] = 200
    img[img==34] = 200
    img[img==35] = 200
    img[img==36] = 200 
    
    

    img[img==37] = 255
    img[img==38] = 255
    img[img==39] = 255
    img[img==40] = 255
    img[img==41] = 255
    img[img==42] = 255
    img[img==43] = 255
    img[img==44] = 255
    img[img==45] = 255
    '''
    
    '''    
    img[img<60] = 0
    img[img>750] = 0
    img = np.uint8(img)
    '''
    img[img<0]=0
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()

    

    #K- Means
    Z = np.reshape(img,(1,262144))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,7,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    cv2.imwrite('res2.jpg',res2)
    #cv2.imwrite('img'+patient_no+'.jpg',pat[int(len(pat)/2)].image)
    img=res2
    img = np.uint8(img)

#binary Conversion
    '''thresh = 127
    im_bw = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
    im_bw=~im_bw
    cv2.imwrite('bw_image.png', im_bw)'''

    

   
    # find surrounding torso from the threshold and make a mask
    im2, contours, _ = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(img.shape, np.uint8)
    cv2.fillPoly(mask, [largest_contour], 255)

    # apply mask to threshold image to remove outside. this is our new mask
    img = ~img
    img[(mask == 0)] = 0 # <-- Larger than threshold value

    # apply closing to the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # <- to remove speckles...
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    
    # apply mask to image
    img2 = pat[int(len(pat)/2)].image.copy()
    img2[(img == 0)] = 0 # <-- Larger than threshold value
    
    print(img2)
    
    plt.imshow(im2, cmap=plt.cm.gray)
    plt.show()
    
    '''
    img2[img2<10] = 0
    img2[img2>600] = 0
    img2 = np.uint8(img2)

    img3 = pat[int(len(pat)/2)].image.copy()
    img3[(img2 == 0)] = 0
    '''
    
    '''
    # closing
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=False)
    largest_contour = max(contours, key=cv2.contourArea)
    rgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    aaa = np.concatenate(sorted_contours[1:3] )
    q=cv2.drawContours(rgb, [cv2.convexHull(aaa)], -1, (0,255,0), 3)
    '''
    #print(img2)

    plt.figure(figsize=(12, 12))
    plt.subplot(131)
    plt.imshow(pat[int(len(pat)/2)].image)
    cv2.imwrite('img0'+'.jpg',pat[int(len(pat)/2)].image)
    #img=~img
    plt.subplot(132)
    plt.imshow(img)
    cv2.imwrite('img1'+'.jpg',img)
    plt.subplot(133)
    plt.imshow(img2)
    cv2.imwrite('img2'+'.jpg',img2)
    plt.show()
