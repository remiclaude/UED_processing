import numpy as np 
import find_peak
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import imutils
import glob
import matplotlib.ticker as ticker
import cv2  # importing cv
from skimage import img_as_float
from skimage import filters
import h5py


# A: original matrix [time x size1 x size2]
# shift: horizontal shift

def shift_0order(A, B, center, s_roi = 20):
    A_shift = np.zeros(np.shape(A))
    s_roi = 20
    for i in range(np.size(A, 0)):
        centerA = find_peak.get_pos_around(A[i], center, s_roi, method='2D_voigt')
        centerB = find_peak.get_pos_around(B[i], center, s_roi, method='2D_voigt')
        shift = centerA-centerB
        f = interpolate.interp2d(np.arange(np.size(A, axis=1)), np.arange(np.size(A, axis=1)), A[i])
        A_shift[i, :, :] = f(np.arange(np.size(A, axis=1))+shift[1], np.arange(np.size(A, axis=2))+shift[0])
    return A_shift 


def shift_0order_regulargrid(A, B, center, s_roi=20):
    A_shift = np.zeros(np.shape(A))
    for i in range(np.size(A, 0)):
        # Define regions of interest (ROI) around the diffraction peak
        centerA = np.reshape(find_peak.get_all(A[i], [center[1]-s_roi, center[1]+s_roi, center[0]-s_roi, center[0]+s_roi])[4:6], (2,))
        centerB = np.reshape(find_peak.get_all(B[i], [center[1]-s_roi, center[1]+s_roi, center[0]-s_roi, center[0]+s_roi])[4:6], (2,))
        
        # Calculate shift needed to align the centers
        shift = centerA - centerB
        
        # Create a meshgrid with the calculated shift
        x = np.arange(np.size(A, axis=1)) + shift[0]
        y = np.arange(np.size(A, axis=2)) + shift[1]
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Define the interpolator with boundary handling
        f = interpolate.RegularGridInterpolator(
            (np.arange(np.size(A, axis=1)), np.arange(np.size(A, axis=2))),
            A[i],
            method='linear',
            bounds_error=False,
            fill_value=0  # Adjust this value if you need a different out-of-bounds behavior
        )
        
        # Interpolate shifted image
        A_shift[i, :, :] = f((X, Y))
        
    return A_shift


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



def shift_0order_fit(A, B, center, s_roi=20, method = '2D_voigt_rotated'):
    A_shift = np.zeros(np.shape(A))
    ROI = [center[1]-s_roi, center[1]+s_roi, center[0]-s_roi, center[0]+s_roi]
    for i in range(np.size(A, 0)):
        # Define regions of interest (ROI) around the diffraction peak
        centerA = (find_peak.get_pos(A[i], ROI, method=method))
        centerB = (find_peak.get_pos(B[i], ROI, method=method))
        
        # Calculate shift needed to align the centers
        shift = centerA - centerB
        
        # Create a meshgrid with the calculated shift
        x = np.arange(np.size(A, axis=1)) + shift[0]
        y = np.arange(np.size(A, axis=2)) + shift[1]
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Define the interpolator with boundary handling
        f = interpolate.RegularGridInterpolator(
            (np.arange(np.size(A, axis=1)), np.arange(np.size(A, axis=2))),
            A[i],
            method='linear',
            bounds_error=False,
            fill_value=np.nan  # Adjust this value if you need a different out-of-bounds behavior
        )
        
        # Interpolate shifted image
        A_shift[i, :, :] = f((X, Y))
        
    return A_shift

def shift_to_ref(A, ref, center, s_roi=20, method = '2D_voigt_rotated'):
    A_shift = np.zeros(np.shape(A))
    ROI = [center[1]-s_roi, center[1]+s_roi, center[0]-s_roi, center[0]+s_roi]
    for i in range(np.size(A, 0)):
        # Define regions of interest (ROI) around the diffraction peak
        centerA = (find_peak.get_pos(A[i], ROI, method=method))
        
        # Calculate shift needed to align the centers
        shift = centerA - ref
        
        # Create a meshgrid with the calculated shift
        x = np.arange(np.size(A, axis=1)) + shift[0]
        y = np.arange(np.size(A, axis=2)) + shift[1]
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Define the interpolator with boundary handling
        f = interpolate.RegularGridInterpolator(
            (np.arange(np.size(A, axis=1)), np.arange(np.size(A, axis=2))),
            A[i],
            method='linear',
            bounds_error=False,
            fill_value=np.nan # Adjust this value if you need a different out-of-bounds behavior
        )
        
        # Interpolate shifted image
        A_shift[i, :, :] = f((X, Y))
        
    return A_shift



def shift_to_ref_cv2(A, ref, center, s_roi=20, method = '2D_voigt_rotated'):
    A_shift = []
    ROI = [center[1]-s_roi, center[1]+s_roi, center[0]-s_roi, center[0]+s_roi]
    for i in range(np.size(A, 0)):
        # Define regions of interest (ROI) around the diffraction peak
        centerA = (find_peak.get_pos(A[i], ROI, method=method))
    
        # Calculate shift needed to align the centers
        shift = centerA - ref
        ### the lower limit for the shift is :
        # shift=np.array([0.05, 0.05])
        ### if lower, the function doesn't do antyhing
        translation_to_center = np.float64([[1, 0, shift[1]],
                                        [0, 1, shift[0]]])
        A_shift.append(cv2.warpAffine(A[i], translation_to_center, (A[i].shape[1], A[i].shape[0]), flags=cv2.INTER_LANCZOS4))
    return np.array(A_shift)


def shift_0order_mass(A, B, center, s_roi = 20):
    A_shift = np.zeros(np.shape(A))
    for i in range(np.size(A, 0)):
        centerA = find_peak.get_center_mass([A[i]], [center[1]-s_roi, center[1]+s_roi, center[0]-s_roi, center[0]+s_roi])[0]
        centerB = find_peak.get_center_mass([B[i]], [center[1]-s_roi, center[1]+s_roi, center[0]-s_roi, center[0]+s_roi])[0]
        shift = centerA-centerB
        f = interpolate.interp2d(np.arange(np.size(A, axis=1)), np.arange(np.size(A, axis=1)), A[i])
        A_shift[i, :, :] = f(np.arange(np.size(A, axis=1))+shift[1], np.arange(np.size(A, axis=2))+shift[0])
    return A_shift 

######################## REMOVE HOT #############################
##### Removing hot pixels from scattering light based on the difference between iOFF and iON
def remove_hot(iON, iOFF, n=2):
    print(np.shape(iON))
    if len(np.shape(iON)) > 2:
        ratio_hot = []
        for o in range(np.shape(iON)[0]):
            diff = np.divide(iON[o], iOFF[o], out=np.ones_like(iON[o]), where=iOFF[o]!=0)
            mean=np.mean(diff)
            s = np.std(diff)
            pic_pos_above = np.where(diff > n)
            pic_pos_below = np.where(diff <= 1/n)
            pic_pos_off = np.where((iOFF[o] == 0) & (iON[o]>n))
            ratio_hot.append(1e2*(len(pic_pos_above[0])+len(pic_pos_below[0])+len(pic_pos_off[0]))/(512*512))
            for i in range(len(pic_pos_off[0])):
                iON[o][pic_pos_off[0][i], pic_pos_off[1][i]] = 0
            for i in range(len(pic_pos_above[0])):
                iON[o][pic_pos_above[0][i], pic_pos_above[1][i]] = iOFF[o][pic_pos_above[0][i], pic_pos_above[1][i]]
            for i in range(len(pic_pos_below[0])):
                iON[o][pic_pos_below[0][i], pic_pos_below[1][i]] = iOFF[o][pic_pos_below[0][i], pic_pos_below[1][i]]
        ratio_hot = np.mean(np.array(ratio_hot))
        print('average ratio of hot pixel : ', ratio_hot) 
    else:
        print('one image processing')
        diff = np.divide(iON, iOFF, out=np.ones_like(iON), where=iOFF!=0)
        print(np.shape(diff))
        pic_pos_above = np.where(diff > n)
        pic_pos_below = np.where(diff < 1/n)
        pic_pos_off = np.where((iOFF == 0) & (iON > n))
        print('number of hot pixel above : ',  len(pic_pos_above[0]))
        print('number of hot pixel below : ', len(pic_pos_below[0]))
        print('number of hot pixel with off = 0 : ', len(pic_pos_off[0]))
        print('ratio of total removed hot pixel: ', 1e2*(len(pic_pos_above[0])+len(pic_pos_below[0])+len(pic_pos_off[0]))/(512*512))
        ratio_hot = 1e2*(len(pic_pos_above[0])+len(pic_pos_below[0])+len(pic_pos_off[0]))/(512*512) 
        for i in range(len(pic_pos_off[0])):
            iON[pic_pos_off[0][i], pic_pos_off[1][i]] = 0
        for i in range(len(pic_pos_above[0])):
            iON[pic_pos_above[0][i], pic_pos_above[1][i]] = iOFF[pic_pos_above[0][i], pic_pos_above[1][i]]
        for i in range(len(pic_pos_below[0])):
            iON[pic_pos_below[0][i], pic_pos_below[1][i]] = iOFF[pic_pos_below[0][i], pic_pos_below[1][i]]

    return iON, iOFF, ratio_hot

##### Removing hot pixels from scattering light based on the difference between iOFF and iON with uneven 
def remove_hot_uneven(iON, iOFF, min, max, plot_show=False):
    print(np.shape(iON))
    if len(np.shape(iON)) > 2:
        ratio_hot = []
        pos_hot = []
        for o in range(np.shape(iON)[0]):
            diff = np.divide(iON[o], iOFF[o], out=np.ones_like(iON[o]), where=iOFF[o]!=0)
            mean=np.mean(diff)
            s = np.std(diff)
            pic_pos_above = np.where(diff > max)
            pic_pos_below = np.where(diff <= min)
            pic_pos_off = np.where((iOFF[o] == 0) & (iON[o]>max))
            # print('length above: ', len(pic_pos_above[0]))
            # print('length below: ', len(pic_pos_below[0]))
            # print('length off: ', len(pic_pos_off[0]))
            pos_hot.append([pic_pos_above, pic_pos_below, pic_pos_off])
            ratio_hot.append(1e2*(len(pic_pos_above[0])+len(pic_pos_below[0])+len(pic_pos_off[0]))/(512*512))
            for i in range(len(pic_pos_off[0])):
                iON[o][pic_pos_off[0][i], pic_pos_off[1][i]] = 0
            for i in range(len(pic_pos_above[0])):
                iON[o][pic_pos_above[0][i], pic_pos_above[1][i]] = iOFF[o][pic_pos_above[0][i], pic_pos_above[1][i]]
            for i in range(len(pic_pos_below[0])):
                iON[o][pic_pos_below[0][i], pic_pos_below[1][i]] = iOFF[o][pic_pos_below[0][i], pic_pos_below[1][i]]
        ratio_hot = np.mean(np.array(ratio_hot))
        print('average ratio of hot pixel : ', ratio_hot) 
        if plot_show:
            img = np.zeros((np.shape(iON)[1], np.shape(iON)[2]))
        # print(pos_hot)
            for hot in pos_hot:
                for h in hot:
                    img[h[0], h[1]] = 1 
            fig, ax = plt.subplots(1,1, figsize=(6,6), layout='tight')
            ax.set_title('position of hot pixel in the map')
            ax.imshow(img, cmap='Greys')
    else:
        print('one image processing')
        diff = np.divide(iON, iOFF, out=np.ones_like(iON), where=iOFF!=0)
        print(np.shape(diff))
        pic_pos_above = np.where(diff > max)
        pic_pos_below = np.where(diff < min)
        pic_pos_off = np.where((iOFF == 0) & (iON > max))
        print('number of hot pixel above : ',  len(pic_pos_above[0]))
        print('number of hot pixel below : ', len(pic_pos_below[0]))
        print('number of hot pixel with off = 0 : ', len(pic_pos_off[0]))
        print('ratio of total removed hot pixel: ', 1e2*(len(pic_pos_above[0])+len(pic_pos_below[0])+len(pic_pos_off[0]))/(512*512))
        pos_hot = [pic_pos_above, pic_pos_below, pic_pos_off]
        ratio_hot = 1e2*(len(pic_pos_above[0])+len(pic_pos_below[0])+len(pic_pos_off[0]))/(512*512) 
        for i in range(len(pic_pos_off[0])):
            iON[pic_pos_off[0][i], pic_pos_off[1][i]] = 0
        for i in range(len(pic_pos_above[0])):
            iON[pic_pos_above[0][i], pic_pos_above[1][i]] = iOFF[pic_pos_above[0][i], pic_pos_above[1][i]]
        for i in range(len(pic_pos_below[0])):
            iON[pic_pos_below[0][i], pic_pos_below[1][i]] = iOFF[pic_pos_below[0][i], pic_pos_below[1][i]]
        if plot_show:
            img = np.zeros((np.shape(iON)[0], np.shape(iON)[1]))
        # print(pos_hot)
            for h in pos_hot:
                img[h[0], h[1]] = 1 
            fig, ax = plt.subplots(1,1, figsize=(6,6), layout='tight')
            ax.imshow(img, cmap='Greys')
    return iON, iOFF, ratio_hot

def get_pos_hot_uneven(iON, iOFF, min, max, plot_show=False):
    print(np.shape(iON))
    if len(np.shape(iON)) > 2:
        ratio_hot = []
        pos_hot = []
        for o in range(np.shape(iON)[0]):
            diff = np.divide(iON[o], iOFF[o], out=np.ones_like(iON[o]), where=iOFF[o]!=0)
            mean=np.mean(diff)
            s = np.std(diff)
            pic_pos_above = np.where(diff > max)
            pic_pos_below = np.where(diff <= min)
            pic_pos_off = np.where((iOFF[o] == 0) & (iON[o]>max))
            # print('length above: ', len(pic_pos_above[0]))
            # print('length below: ', len(pic_pos_below[0]))
            # print('length off: ', len(pic_pos_off[0]))
            pos_hot.append([pic_pos_above, pic_pos_below, pic_pos_off])
            ratio_hot.append(1e2*(len(pic_pos_above[0])+len(pic_pos_below[0])+len(pic_pos_off[0]))/(512*512))
            for i in range(len(pic_pos_off[0])):
                iON[o][pic_pos_off[0][i], pic_pos_off[1][i]] = 0
            for i in range(len(pic_pos_above[0])):
                iON[o][pic_pos_above[0][i], pic_pos_above[1][i]] = iOFF[o][pic_pos_above[0][i], pic_pos_above[1][i]]
            for i in range(len(pic_pos_below[0])):
                iON[o][pic_pos_below[0][i], pic_pos_below[1][i]] = iOFF[o][pic_pos_below[0][i], pic_pos_below[1][i]]
        ratio_hot = np.mean(np.array(ratio_hot))
        print('average ratio of hot pixel : ', ratio_hot) 
        if plot_show:
            img = np.zeros((np.shape(iON)[1], np.shape(iON)[2]))
        # print(pos_hot)
            for hot in pos_hot:
                for h in hot:
                    img[h[0], h[1]] = 1 
            fig, ax = plt.subplots(1,1, figsize=(6,6), layout='tight')
            ax.imshow(img, cmap='Greys')
    else:
        print('one image processing')
        diff = np.divide(iON, iOFF, out=np.ones_like(iON), where=iOFF!=0)
        print(np.shape(diff))
        pic_pos_above = np.where(diff > max)
        pic_pos_below = np.where(diff < min)
        pic_pos_off = np.where((iOFF == 0) & (iON > max))
        print('number of hot pixel above : ',  len(pic_pos_above[0]))
        print('number of hot pixel below : ', len(pic_pos_below[0]))
        print('number of hot pixel with off = 0 : ', len(pic_pos_off[0]))
        print('ratio of total removed hot pixel: ', 1e2*(len(pic_pos_above[0])+len(pic_pos_below[0])+len(pic_pos_off[0]))/(512*512))
        pos_hot = [pic_pos_above, pic_pos_below, pic_pos_off]
        ratio_hot = 1e2*(len(pic_pos_above[0])+len(pic_pos_below[0])+len(pic_pos_off[0]))/(512*512) 
        for i in range(len(pic_pos_off[0])):
            iON[pic_pos_off[0][i], pic_pos_off[1][i]] = 0
        for i in range(len(pic_pos_above[0])):
            iON[pic_pos_above[0][i], pic_pos_above[1][i]] = iOFF[pic_pos_above[0][i], pic_pos_above[1][i]]
        for i in range(len(pic_pos_below[0])):
            iON[pic_pos_below[0][i], pic_pos_below[1][i]] = iOFF[pic_pos_below[0][i], pic_pos_below[1][i]]
        if plot_show:
            img = np.zeros((np.shape(iON)[0], np.shape(iON)[1]))
        # print(pos_hot)
            for h in pos_hot:
                img[h[0], h[1]] = 1 
            fig, ax = plt.subplots(1,1, figsize=(6,6), layout='tight')
            ax.set_title('position of hot pixel in the map')
            ax.imshow(img, cmap='Greys')
    return iON, iOFF, img, ratio_hot


def remove_hot_abs(iON, iOFF, thresh, plot_show=False):
    print(np.shape(iON))
    if len(np.shape(iON)) > 2:
        ratio_hot = []
        pos_hot = []
        for o in range(np.shape(iON)[0]):
            pic_pos_above_on = np.where(iON[o] > thresh)
            pic_pos_above_off = np.where(iOFF[o] > thresh)
            pos_hot.append([pic_pos_above_on, pic_pos_above_off])
            ratio_hot.append(1e2*(len(pic_pos_above_off[0])+len(pic_pos_above_on[0]))/(512*512))
            for i in range(len(pic_pos_above_on[0])):
                iON[o][pic_pos_above_on[0][i], pic_pos_above_on[1][i]] = 0
            for i in range(len(pic_pos_above_off[0])):
                iOFF[o][pic_pos_above_off[0][i], pic_pos_above_off[1][i]] = 0
        ratio_hot = np.mean(np.array(ratio_hot))
        print('average ratio of hot pixel : ', ratio_hot) 

        if plot_show:
            img = np.zeros((np.shape(iON)[1], np.shape(iON)[2]))
            # print(pos_hot)
            for hot in pos_hot:
                for h in hot:
                    img[h[0], h[1]] = 1 
            fig, ax = plt.subplots(1,1, figsize=(6,6), layout='tight')
            ax.set_title('position of hot pixel in the map')
            ax.imshow(img, cmap='Greys')
    else:
        print('one image processing')
        pic_pos_above_off = np.where(iON >thresh)
        pic_pos_above_off = np.where(iOFF < thresh)
        print('number of hot pixel above : ',  len(pic_pos_above_on[0]))
        print('number of hot pixel below : ', len(pic_pos_above_off[0]))
        print('ratio of total removed hot pixel: ', 1e2*(len(pic_pos_above_on[0])+len(pic_pos_above_off[0]))/(512*512))
        ratio_hot = 1e2*(len(pic_pos_above_on[0])+len(pic_pos_above_off[0]))/(512*512) 
        for i in range(len(pic_pos_above_off[0])):
            iOFF[pic_pos_above_off[0][i], pic_pos_above_off[1][i]] = 0
        for i in range(len(pic_pos_above_on[0])):
            iON[pic_pos_above_on[0][i], pic_pos_above_on[1][i]] = 0
    return iON, iOFF, ratio_hot

################## REMOVE HOT FOR GAUSSIAN DISTRIBUTION #########################
def remove_hot_gauss(iON, iOFF, n=3):
    ratio_hot = []
    for o in range(np.shape(iON)[0]):
        diff = np.divide(iON[o], iOFF[o], out=np.ones_like(iON[o]), where=iOFF[o]!=0)
        mean=np.mean(diff)
        s = np.std(diff)
        try:
            pic_pos_above = np.where(diff>(mean+n*s))
            pic_pos_below = np.where(diff<(mean-n*s))
        except:
            pass
        ratio_hot.append(1e2*(len(pic_pos_above[0])+len(pic_pos_below[0]))/(512*512))
        for i in range(len(pic_pos_above[0])):
            iON[o][pic_pos_above[0][i], pic_pos_above[1][i]] = iOFF[o][pic_pos_above[0][i], pic_pos_above[1][i]]
        for i in range(len(pic_pos_below[0])):
            iON[o][pic_pos_below[0][i], pic_pos_below[1][i]] = iOFF[o][pic_pos_below[0][i], pic_pos_below[1][i]]
    print('average ratio of hot pixel : ', np.mean(np.array(ratio_hot))) 
    return iON, iOFF



def update_pos_oneBP(img, bp, s_roi=5, method = '2D_gaussian', show_plot=False):
    """""
        Here we update one peak position for one or several image
        The function takes:
            -  one array of shape (n_delay, x, y) or (x, y) with x, y the size of one diffraction patterns (usually 512*512)
            -  An array of estimation of the Bragg peak position (2,) size
            -  The size of region of interest to make the fit 
            -  The method of fiting 
        Return: 
            -  the updated position from the fit
    """""
    if len(np.shape(img)) == 3 :
        pos = []
        ROI = [bp[1]-s_roi, bp[1]+s_roi, bp[0]-s_roi, bp[0]+s_roi]
        for k in range(np.shape(img)[0]):
            pos.append(find_peak.get_pos(img[k], ROI, method, show_plot))
        pos = np.array(pos)
    elif len(np.shape(img)) == 2: 
        ROI = [bp[1]-s_roi, bp[1]+s_roi, bp[0]-s_roi, bp[0]+s_roi]
        pos = (find_peak.get_pos(img, ROI, method, show_plot))
    else:
        print('error in get_distance of functions_analysis, not the proper shape')
        pos = False
    return pos 

def update_pos(img, BP, s_roi, method= '2D_gaussian', show_plot=False):
    """""
        Here we update one or several Bragg peak position for one or several image
        The function takes:
            -  one array of shape (n_delay, x, y) or (x, y) with x, y the size of one diffraction patterns (usually 512*512)
            -  An array of estimation of the Bragg peak position (2,) size
            -  The size of region of interest to make the fit 
            -  The method of fiting 
        Return: 
            -  the updated position from the fit either as list for different BP or whatever update_pos_onBP return (array or (n,2) or (2,))
    """""
    if len(np.shape(BP)) == 1:
        return update_pos_oneBP(img, BP, s_roi, method, show_plot)
    elif len(np.shape(BP)) == 2: 
        BP_update = [] 
        for bp in BP:
            BP_update.append(update_pos_oneBP(img, bp, s_roi, method, show_plot))
        return BP_update
    else:
        print('wrong argument in upedate_pos of functions analysis ')
        return False


def flip_with_respect_to_point(image, flip_point, flip_axis=0):
    """
    Flip a grayscale image with respect to a specified floating-point center.

    Args:
        image (np.ndarray): Input grayscale image (NumPy array) with shape (512, 512).
        flip_point (tuple): The point (x, y) with respect to which the image will be flipped.
        flip_axis (int): Axis along which to flip: 0 for vertical, 1 for horizontal.

    Returns:
        np.ndarray: The flipped image.
    """
    image_center = (image.shape[1] / 2  - 0.5, image.shape[0] / 2 - 0.5)
    
    # Step 1: Translate the image to center the flip point at the origin
    translation_to_center = np.float32([[1, 0, image_center[1] - flip_point[1]],
                                        [0, 1, image_center[0] - flip_point[0]]])
    centered_image = cv2.warpAffine(image, translation_to_center, (image.shape[1], image.shape[0]))
    print('image center : ', image_center)
    print('translation matrix : ', translation_to_center)
    print('0th order after translation : ', find_peak.get_pos_around(centered_image, np.round(image_center).astype(np.int16), 15, '2D_voigt_rotated', False))
    # Step 2: Flip the centered image
    if flip_axis == 0:  # Vertical flip
        flipped_image = cv2.flip(centered_image, 0)
    elif flip_axis == 1:  # Horizontal flip
        flipped_image = cv2.flip(centered_image, 1)
    else:
        raise ValueError("flip_axis must be 0 (vertical) or 1 (horizontal).")

    # Step 3: Translate the image back to the original position
    translation_back = np.float32([[1, 0, flip_point[1] - image_center[1]],
                                   [0, 1, flip_point[0] - image_center[0]]])
    final_image = cv2.warpAffine(flipped_image, 
    translation_back, (image.shape[1], image.shape[0]))
    print('0th order after translation back : ', find_peak.get_pos_around(final_image, np.round(flip_point).astype(np.int16), 15, '2D_voigt_rotated', False))

    return final_image

def get_distance(img, BP, s_roi=5, method='2D_gaussian', show_plot=False):
    """""
        Here we get the distance between peaks for only one image
        The function takes:
            -  one array of shape (n_delay, x, y) or (x, y) with x, y the size of one diffraction patterns (usually 512*512)
            -  An array of estimation of the Bragg peak position 2*2 size
            -  The size of region of interest to make the fit 
            -  The method of fiting 
        Return: 
            -  vector between the peak either shape of (n_delay, 2) or (2,)
    """""
    if len(np.shape(img)) == 3 :
        pos_delay = []
        for bp in BP:
            pos_array= []
            ROI = [bp[1]-s_roi, bp[1]+s_roi, bp[0]-s_roi, bp[0]+s_roi]
            for k in range(np.shape(img)[0]):
                pos_array.append(find_peak.get_pos(img[k], ROI, method, show_plot))
            pos_delay.append(np.array(pos_array))
        pos_delay = np.array(pos_delay)
        vec = pos_delay[0, :,:]-pos_delay[1,:,:]
    elif len(np.shape(img)) == 2: 
        pos= []
        for bp in BP:
            ROI = [bp[1]-s_roi, bp[1]+s_roi, bp[0]-s_roi, bp[0]+s_roi]
            pos.append(find_peak.get_pos(img, ROI, method, show_plot))
        pos = np.array(pos)
        vec = pos[0]-pos[1]
    else:
        print('error in get_distance of functions_analysis, not the proper shape')
        vec = False
    return vec 


def get_distance_on_off(imgON, imgOFF, BP, s_roi=5, method = '2D_gaussian', show_plot=False):
    """""
        Here we get the distance between peaks for ON and OFF img
        The function takes:
            -  two arrays of shape (n_delay, x, y) with x, y the size of one diffraction patterns (usually 512*512)
            -  An array of estimation of the Bragg peak position 2*2 size
            -  The size of region of interest to make the fit 
            -  The method of fiting 
        Return: 
            -  vector between the peak for ON and for OFF
    """""
    if show_plot and len(np.shape(imgON))==2:
        vecON = get_distance(imgON, BP, s_roi, method, True)
        vecOFF = get_distance(imgOFF, BP, s_roi, method, True) ## only plot for on, otherwise too much figure
    else:
        vecON = get_distance(imgON, BP, s_roi, method, False)
        vecOFF = get_distance(imgOFF, BP, s_roi, method, False)

    if show_plot and len(np.shape(imgON))>2: 
        BP_update = update_pos(imgON[0], BP, s_roi, method, True)

        fig, axs = plt.subplots(1, 2, figsize=(14,7), layout='tight')
        fig.suptitle('Distance between two Bragg peak with ' + method)
        ax = axs[0]
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        im_plot = ax.imshow(imgON[0], cmap = 'bwr', vmin = 0, vmax = np.nanpercentile(imgON[0], 99.9))
        fig.colorbar(im_plot, cax=cax, shrink=0.1, orientation='vertical', label = r'electron count variation ($\%$)', format='%.1e', location='right')

        for bp in BP_update:
            ax.add_patch(Circle((bp[1], bp[0]) , radius = 5,alpha=1, fill = False, edgecolor = 'k', lw = 1))
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axs[1]
        ax.plot(np.arange(np.shape(vecON)[0]), np.linalg.norm(vecON, axis=1), '+--', c='r', lw=1 , label = 'pump on')
        ax.plot(np.arange(np.shape(vecOFF)[0]), np.linalg.norm(vecOFF, axis=1) , '+--', c= 'b', lw=1,  label = 'pump off')
        ax_twin = ax.twinx()
        ax_twin.plot(np.arange(np.shape(vecON)[0]), np.linalg.norm(vecON, axis=1)-np.linalg.norm(vecOFF, axis=1), 'o-', c='k', lw=2, label='on-off' )
        ax_twin.set_ylabel('distance variation (px)')
        ax.legend(loc='upper left')
        ax_twin.legend(loc='upper right')
        ax.set_xlabel('delay position')
        ax.set_ylabel('distance between peaks (px)')
    return vecON, vecOFF


########## ROTATE VECTOR @############
## rotation of a vector by a given angle in radian

def rotate_vector(vector, angle):
    x = vector[0] * np.cos(angle) - vector[1] * np.sin(angle)
    y = vector[0] * np.sin(angle) + vector[1] * np.cos(angle)
    return np.array([x, y])



######################## FMT ###########################
### Format for scienitfic notation
def fmt(x, pos):
    a, b = '{:.0e}'.format(x).split('e')
    b = int(b)
    return r'${} \cdot 10^{{{}}}$'.format(a, b)



###################### GET COUNT #########################
def get_count(imgON, imgOFF, bp, radius = 4.5, adjust_pos = True, save='Roi'):    
    if adjust_pos:
        bp = bp.astype(np.int64)
        s_roi = 10
        pos = []
        for i in range(len(bp)):
            pos.append(find_peak.get_pos_around(imgON[0], bp[i], s_roi, '2D_voigt_rotated'))
    else:
        pos = bp
        print('no find peak')

    filterK = np.zeros(np.shape(imgOFF[0]), dtype=float)
    for k in range(len(pos)):
        for i in range(np.shape(imgOFF[0])[0]):
            for j in range(np.shape(imgOFF[0])[1]): 
                if np.linalg.norm(np.array([i,j]) - pos[k]) <= radius:
                    filterK[i,j] = 1
    img_roi_ON_rotated_filterK = filterK*imgON
    img_roi_OFF_rotated_filterK = filterK*imgOFF
    sum_roi = np.nansum(img_roi_ON_rotated_filterK, axis=(1,2))/np.nansum(img_roi_OFF_rotated_filterK, axis=(1,2))
    # print(np.shape(sum_roi))
    return sum_roi

def get_count_ON(imgON, bp, radius = 4.5, adjust_pos_once = True, adjust_pos_continuous = False, show_plot=False):  
    if adjust_pos_once:
        bp = bp.astype(np.int64)
        s_roi = 10
        pos = []
        for i in range(len(bp)):
            pos.append(find_peak.get_pos_around(imgON[0], bp[i], s_roi, '2D_voigt_rotated', show_plot))
    else:
        pos = bp
        print('no find peak')

    if adjust_pos_continuous:
        filter_all = []
        filter_one = np.zeros(np.shape(imgON[0]), dtype=float)
        for im in imgON:
            for k in range(len(pos)):
                pos_update = find_peak.get_pos_around(im, pos[k], s_roi, '2D_voigt_rotated', False)
                for i in range(np.shape(imgON[0])[0]):
                    for j in range(np.shape(imgON[0])[1]): 
                        if np.linalg.norm(np.array([i,j]) - pos_update[k]) <= radius:
                            filter_one[i,j] = 1
            filter_all.append(filter_one)
        filter_all = np.array(filter_all)
        img_roi_ON_rotated_filterK = filter_all*imgON
        sum_roi = np.nansum(img_roi_ON_rotated_filterK, axis=(1,2))
    else:
        filter_one = np.zeros(np.shape(imgON[0]), dtype=float)
        for k in range(len(pos)):
            for i in range(np.shape(imgON[0])[0]):
                for j in range(np.shape(imgON[0])[1]): 
                    if np.linalg.norm(np.array([i,j]) - pos[k]) <= radius:
                        filter_one[i,j] = 1
        img_roi_ON_rotated_filterK = filter_one*imgON
        sum_roi = np.nansum(img_roi_ON_rotated_filterK, axis=(1,2))
    # print(np.shape(sum_roi))
    return sum_roi


def extract_parameter(param_list, param_name):
    return np.array([param[param_name].value for param in param_list])

def fit_BP(img, bp, s_roi):
    bp = bp.astype(np.int64)
    out = find_peak.get_all(img, [bp[1]-s_roi, bp[1]+s_roi, bp[0]-s_roi, bp[0]+s_roi])
    pos= (np.reshape(np.array(out[4:6]), (2,)))
    height = (np.reshape(np.array(out[2:4]), (2,)))
    sigma = (np.reshape(np.array(out[6:8]), (2,)))
    return pos, height, sigma


###################### GET COUNT 1 image #########################
def get_count_1img(imgON, bp, radius = 4.5, adjust_pos = True, save='Roi'):    
    if adjust_pos:
        bp = bp.astype(np.int64)
        s_roi = 10
        pos = []
        for i in range(len(bp)):
            pos.append(np.reshape(np.array(find_peak.get_all(np.sum(imgON, axis=0), [bp[i][1]-s_roi, bp[i][1]+s_roi, bp[i][0]-s_roi, bp[i][0]+s_roi])[4:6]), (2,)))
    else:
        pos = bp
        print('no find peak')

    filterK = np.zeros(np.shape(imgON[0]), dtype=float)
    for k in range(len(pos)):
        for i in range(np.shape(imgON[0])[0]):
            for j in range(np.shape(imgON[0])[1]): 
                if np.linalg.norm(np.array([i,j]) - pos[k]) <= radius:
                    filterK[i,j] = 1
    img_roi_ON_rotated_filterK = filterK*imgON
    sum_roi = np.nansum(img_roi_ON_rotated_filterK, axis=(1,2))
    # print(np.shape(sum_roi))
    return sum_roi


###################### GET COUNT ARC #########################
def get_count_arc(imgON, imgOFF, bp, r1 = 4.5, r2 = 10, adjust_pos = True, save='Roi'):    
    if adjust_pos:
        bp = bp.astype(np.int64)
        s_roi = 10
        pos = []
        for i in range(len(bp)):
            pos.append(np.reshape(np.array(find_peak.get_all(np.sum(imgOFF, axis=0), [bp[i][1]-s_roi, bp[i][1]+s_roi, bp[i][0]-s_roi, bp[i][0]+s_roi])[4:6]), (2,)))
    else:
        pos = bp
        print('no find peak')

    filterK = np.zeros(np.shape(imgOFF[0]), dtype=float)
    for k in range(len(pos)):
        for i in range(np.shape(imgOFF[0])[0]):
            for j in range(np.shape(imgOFF[0])[1]): 
                if np.linalg.norm(np.array([i,j]) - pos[k]) <= r2 and np.linalg.norm(np.array([i,j]) - pos[k]) >= r1:
                    filterK[i,j] = 1
    img_roi_ON_rotated_filterK = filterK*imgON
    img_roi_OFF_rotated_filterK = filterK*imgOFF
    sum_roi = np.nansum(img_roi_ON_rotated_filterK, axis=(1,2))/np.nansum(img_roi_OFF_rotated_filterK, axis=(1,2))
    # print(np.shape(sum_roi))
    return sum_roi


###################### BUILD BP_POS #########################
def bragg_peak_position(Zorder, L, sample='graphite'):
    if sample=='graphite':
        BP_BZ = []
        for i in range(6):
            BP_BZ.append([L*np.cos(np.pi*i/3), L*np.sin(np.pi*i/3)])
        BP_pos = [Zorder + np.array(BP_BZ)]
        BP_BZ = []
        for i in range(6):
            BP_BZ.append([2*L*np.cos(np.pi*i/3), 2*L*np.sin(np.pi*i/3)])
        BP_pos.append(Zorder + np.array(BP_BZ))
        BP_BZ = []
        for i in range(6):
            BP_BZ.append([np.sqrt(3)*L*np.cos(np.pi*i/3+np.pi/6), np.sqrt(3)*L*np.sin(np.pi*i/3+np.pi/6)])
        BP_pos.append(Zorder + np.array(BP_BZ))
        BP_BZ = []
    elif sample=='BSCCO':
        BP_BZ = []
        BP_BZ_2 = []
        for i in range(2):
            BP_BZ.append([L*np.cos(np.pi*i), L*np.sin(np.pi*i)])
            BP_BZ_2.append([2*L*np.cos(np.pi*i), 2*L*np.sin(np.pi*i)])
        BP_pos = [Zorder + np.array(BP_BZ)]
        BP2_pos =[Zorder + np.array(BP_BZ_2)]

        BP_BZ = []
        BP_BZ_2 = []
        for i in range(2):
            BP_BZ.append([L*np.cos(np.pi*i+np.pi/2), L*np.sin(np.pi*i+np.pi/2)])
            BP_BZ_2.append([2*L*np.cos(np.pi*i+np.pi/2), 2*L*np.sin(np.pi*i+np.pi/2)])
        BP_pos.append(Zorder + np.array(BP_BZ))
        BP2_pos.append(Zorder + np.array(BP_BZ_2))

        BP_BZ = []
        BP_BZ_2 = []
        for i in range(4):
            BP_BZ.append([np.sqrt(2)*L*np.cos(np.pi/4+np.pi*i/2), np.sqrt(2)*L*np.sin(np.pi/4+np.pi*i/2)])
            BP_BZ_2.append([2*np.sqrt(2)*L*np.cos(np.pi/4+np.pi*i/2), 2*np.sqrt(2)*L*np.sin(np.pi/4+np.pi*i/2)])

        BP_pos.append(Zorder + np.array(BP_BZ))
        BP2_pos.append(Zorder + np.array(BP_BZ_2))
    return BP_pos

###################### BUILD DS_POS #########################
def diffuse_scattering_position(Zorder, L):
    edge = L*np.tan(np.pi/6)
    BP_DS = []
    for i in range(6):
        BP_DS.append([2*edge*np.cos(np.pi*i/3+np.pi/6), 2*edge*np.sin(np.pi*i/3+np.pi/6)])
    DS_pos = [Zorder + np.array(BP_DS)]
    BP_DS = []
    for i in range(6):
        BP_DS.append([4*edge*np.cos(np.pi*i/3+np.pi/6), 4*edge*np.sin(np.pi*i/3+np.pi/6)])
    DS_pos.append(Zorder + np.array(BP_DS))
    BP_DS = []
    for i in range(6):  
        BP_DS.append(rotate_vector(np.array([3*L/2, edge/2]), i*np.pi/3))
        BP_DS.append(rotate_vector(np.array([3*L/2,-edge/2]), i*np.pi/3))
    DS_pos.append(Zorder + np.array(BP_DS))
    BP_DS = []
    for i in range(6):
        BP_DS.append(rotate_vector(np.array([2*L, edge]), i*np.pi/3))
        BP_DS.append(rotate_vector(np.array([2*L,-edge]), i*np.pi/3))
    DS_pos.append(Zorder + np.array(BP_DS))
    BP_DS = []
    return DS_pos


def diffuse_scattering_position2(Zorder, L):
    edge = L*np.tan(np.pi/6)
    BP_DS = []
    for i in range(6):
        BP_DS.append([2*edge*np.cos(np.pi*i/3+np.pi/6), 2*edge*np.sin(np.pi*i/3+np.pi/6)])
    DS_pos = [Zorder + np.array(BP_DS)]
    BP_DS = []
    # for i in range(6):
    #     BP_DS.append([4*edge*np.cos(np.pi*i/3+np.pi/6), 4*edge*np.sin(np.pi*i/3+np.pi/6)])
    # DS_pos.append(Zorder + np.array(BP_DS))
    # BP_DS = []
    for i in range(6):  
        BP_DS.append(rotate_vector(np.array([edge/2, np.sqrt(3)*L]), i*np.pi/3))
        BP_DS.append(rotate_vector(np.array([-edge/2, np.sqrt(3)*L]), i*np.pi/3))
    DS_pos.append(Zorder + np.array(BP_DS))
    # BP_DS = []
    # for i in range(6):
    #     BP_DS.append(rotate_vector(np.array([2*L, edge]), i*np.pi/3))
    #     BP_DS.append(rotate_vector(np.array([2*L,-edge]), i*np.pi/3))
    # DS_pos.append(Zorder + np.array(BP_DS))
    BP_DS = []
    return DS_pos