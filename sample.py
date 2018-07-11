import numpy as np
import pydicom
import os
import scipy.ndimage

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import cv2

import bestfit

folder='test_brain'

def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    return slices


def rescale_correction(s):
    return  s* 1 - 1024
    
  
def hu_2_pixels(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    image[image == -2000] = 0
    for slice_number in range(len(slices)):
        image[slice_number]=rescale_correction(image[slice_number])
        image[slice_number]=bestfit.imagecropper(image[slice_number])
    return np.array(image, dtype=np.int16)


def resample(pixels, slices, new_spacing=[1,1,1]):
    spacing=np.array([slices[0].SliceThickness]+list(slices[0].PixelSpacing),dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = pixels.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / pixels.shape
    new_spacing = spacing / real_resize_factor
    
    
    pixels = scipy.ndimage.interpolation.zoom(pixels, real_resize_factor, mode='nearest')
    
    return pixels, new_spacing


def plot_3d(pixels):
    
    pixels=pixels.transpose(2,1,0)
    verts, faces, normals, values = measure.marching_cubes_lewiner(pixels, 53)
   
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, pixels.shape[0])
    ax.set_ylim(0, pixels.shape[1])
    ax.set_zlim(0, pixels.shape[2])

    plt.show()

def imagesee(pixels):
    cv2.imshow('asda',pixels)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
slices=load_scan(folder)
pixels=hu_2_pixels(slices)
print("Converted")
#pix_resampled, spacing = resample(pixels, slices, [1,1,1])
print("Sampled")
plot_3d(pixels)







