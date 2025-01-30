
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
import time
from random import seed
from random import random
from IPython.display import clear_output
from matplotlib.pyplot import figure, draw, pause
from PIL import Image, ImageEnhance, ImageFilter
from numpy import *
from scipy import optimize
from scipy.optimize import curve_fit, minimize
import glob
import itertools
from matplotlib.widgets import RectangleSelector
import os.path
from matplotlib import rcParams
from scipy import ndimage
import matplotlib.pyplot as plt
# rcParams['text.latex.preamble'] = [r"\usepackage{physics}"]
# rcParams['text.usetex'] = True
# rcParams['font.family'] = 'serif'
# rcParams['font.size'] = 10
import lmfit
from lmfit.models import (ConstantModel, GaussianModel, LinearModel, LorentzianModel, QuadraticModel, VoigtModel, Gaussian2dModel)
from scipy.interpolate import griddata
from lmfit.lineshapes import gaussian2d, lorentzian
from scipy.interpolate import griddata
from scipy.special import wofz
from matplotlib.patches import Circle

def moments_gauss(data, x):
    offset = np.min(data)
    height = np.max(data) - offset
    HM = offset + height / 2
    over = np.where(data > HM)[0]
    fwhm = x[over[-1]] - x[over[0]]
    center = x[np.argmax(data)]
    sigma = fwhm / 2.
    gamma = sigma
    amplitude = height * sigma / 0.3989423
    return amplitude, center, sigma, gamma, offset


def fit_gauss(data, x):
    model = GaussianModel() + QuadraticModel()
    initial_guess = moments_gauss(data, x)
    params = model.make_params(amplitude=initial_guess[0], center=initial_guess[1], sigma=initial_guess[2], gamma=initial_guess[3], c=initial_guess[4], b=0, a=0)
    y = data
    result = model.fit(y, params, x=x, max_nfev=400)
    print(params)
    return result


# Define the 2D Voigt function
def voigt_2d(x, y, amplitude, centerx, centery, sigmax, sigmay, gammax, gammay, offset):
    """
    2D Voigt function, with independent Gaussian and Lorentzian components for x and y directions.

    Parameters:
    - x, y: meshgrid of independent variables
    - amplitude: peak height
    - x0, y0: center of the peak in x and y
    - sigma_x, sigma_y: standard deviations of the Gaussian components in x and y
    - gamma_x, gamma_y: half-width at half-maximum (HWHM) of the Lorentzian components in x and y
    """
    # Convert x and y to complex form for Voigt profile along each axis
    z_x = ((x - centerx) + 1j * gammax) / (sigmax * np.sqrt(2))
    z_y = ((y - centery) + 1j * gammay) / (sigmay * np.sqrt(2))

    # Compute the Voigt profile separately along x and y, then take the product
    voigt_x = np.real(wofz(z_x)) / (sigmax * np.sqrt(2 * np.pi))
    voigt_y = np.real(wofz(z_y)) / (sigmay * np.sqrt(2 * np.pi))
    
    # The full 2D Voigt profile as the product of independent x and y Voigt profiles
    return offset + amplitude * voigt_x * voigt_y

# Fit function to 2D data using the Voigt profile with lmfit
def fit_voigt_2d(x_data, y_data, z_data):
    """
    Fits a 2D dataset to a Voigt profile with independent x and y Gaussian/Lorentzian components.

    Parameters:
    - x_data: 2D numpy array for x values
    - y_data: 2D numpy array for y values
    - z_data: 2D numpy array for the data (dependent variable)

    Returns:
    - result: lmfit ModelResult containing fit parameters and statistics
    """
    # Create a model from the custom Voigt function
    voigt_model = lmfit.Model(voigt_2d, independent_vars=['x', 'y'])
    # Initialize parameters for the fit
    params = voigt_model.make_params(
        amplitude=dict(value=np.max(z_data),min=0, max=+inf) ,
        centerx= dict(value=np.mean(x_data), min=x_data[0, 0], max=x_data[-1, -1]),
        centery=dict(value=np.mean(y_data), min=y_data[0,0], max=y_data[-1,-1]),
        sigmax=dict(value=1.0, min=0, max=10),
        sigmay=dict(value=1.0, min=0, max=10),
        gammax=dict(value=1.0, min=0, max=1e3),
        gammay=dict(value=1.0, min=0, max=1e3),
        offset = dict(value=np.min(z_data), min=0, max=+inf)
    )

    # Perform the fit (flattening data to 1D for fitting)
    result = voigt_model.fit(z_data.ravel(), x=x_data.ravel(), y=y_data.ravel(), params=params)
    return result, voigt_model


def calculate_fwhm_voigt(sigma_x, sigma_y, gamma_x, gamma_y):
    # Gaussian FWHM
    fwhm_gaussian_x = 2 * np.sqrt(2 * np.log(2)) * sigma_x
    fwhm_gaussian_y = 2 * np.sqrt(2 * np.log(2)) * sigma_y
    
    # Lorentzian FWHM
    fwhm_lorentzian_x = 2 * gamma_x
    fwhm_lorentzian_y = 2 * gamma_y
    
    # Voigt FWHM
    fwhm_voigt_x = (0.5346 * fwhm_lorentzian_x +
                    np.sqrt(0.2166 * fwhm_lorentzian_x**2 + fwhm_gaussian_x**2))
    fwhm_voigt_y = (0.5346 * fwhm_lorentzian_y +
                    np.sqrt(0.2166 * fwhm_lorentzian_y**2 + fwhm_gaussian_y**2))
    
    return fwhm_voigt_x, fwhm_voigt_y

# Define the rotated 2D Voigt function
def voigt_2d_rotated(x, y, amplitude, centerx, centery, sigmax, sigmay, gammax, gammay, theta, offset):
    """
    2D Voigt function with rotation.

    Parameters:
    - x, y: meshgrid of independent variables
    - amplitude: peak height
    - x0, y0: center of the peak in x and y
    - sigma_x, sigma_y: standard deviations of the Gaussian components in x and y
    - gamma_x, gamma_y: half-width at half-maximum (HWHM) of the Lorentzian components in x and y
    - theta: rotation angle (in radians) of the peak

    Returns:
    - 2D Voigt profile with rotation applied
    """
    # Apply rotation transformation
    x_rot = (x - centerx) * np.cos(theta) + (y - centery) * np.sin(theta)
    y_rot = -(x - centerx) * np.sin(theta) + (y - centery) * np.cos(theta)

    # Convert rotated x and y to complex form for Voigt profile along each axis
    z_x = (x_rot + 1j * gammax) / (sigmax * np.sqrt(2))
    z_y = (y_rot + 1j * gammay) / (sigmay * np.sqrt(2))

    # Compute the Voigt profile separately along x_rot and y_rot, then take the product
    voigt_x = np.real(wofz(z_x)) / (sigmax * np.sqrt(2 * np.pi))
    voigt_y = np.real(wofz(z_y)) / (sigmay * np.sqrt(2 * np.pi))
    
    # The full 2D rotated Voigt profile
    return offset + amplitude * voigt_x * voigt_y

# Fit function to 2D data using the rotated Voigt profile with lmfit
def fit_voigt_2d_rotated(x_data, y_data, z_data):
    """
    Fits a 2D dataset to a rotated Voigt profile.

    Parameters:
    - x_data: 2D numpy array for x values
    - y_data: 2D numpy array for y values
    - z_data: 2D numpy array for the data (dependent variable)

    Returns:
    - result: lmfit ModelResult containing fit parameters and statistics
    """
    # Create a model from the custom rotated Voigt function
    voigt_model = lmfit.Model(voigt_2d_rotated, independent_vars=['x', 'y'])

    # Initialize parameters for the fit
    params = voigt_model.make_params(
        amplitude=dict(value=np.max(z_data),min=0, max=+inf) ,
        centerx= dict(value=np.mean(x_data), min=x_data[0, 0], max=x_data[-1, -1]),
        centery=dict(value=np.mean(y_data), min=y_data[0,0], max=y_data[-1,-1]),
        sigmax=dict(value=1.0, min=0, max=10),
        sigmay=dict(value=1.0, min=0, max=10),
        gammax=dict(value=0.9, min=0, max=1e3),
        gammay=dict(value=0.9, min=0, max=1e3),
        theta=dict(value=2, min=0, max=+np.pi*2),
        offset = dict(value=np.min(z_data), min=0, max=+inf)
    )
    # print(1/np.sqrt(z_data))
    # Fit the model to the data (flattening data to 1D for fitting)
    result = voigt_model.fit(z_data.ravel(), x=x_data.ravel(), y=y_data.ravel(), params=params, weights=1/np.sqrt(1+z_data).ravel())
    # print(result.fit_report())
    return result, voigt_model


def fit_2dvoigt(img, ROI=None, show_plot=False, rotation=False):
    if len(np.shape(img))>2:
        x = np.arange(np.size(img, axis=1)) 
        y = np.arange(np.size(img, axis=2)) 
        X, Y = np.meshgrid(x, y, indexing='ij')
    else:
        if ROI!=None:
            A = make_ROI(img,*ROI)
            x = np.arange(np.size(A, axis=0))+ROI[2] 
            y = np.arange(np.size(A, axis=1))+ROI[0] 
        else: 
            A = img
            x = np.arange(np.size(A, axis=0)) 
            y = np.arange(np.size(A, axis=1)) 
        ## create axis of ROI
        ## create meshgrid
        X, Y = np.meshgrid(x, y, indexing='ij')
        if rotation:
            result, model = fit_voigt_2d_rotated(X, Y, A)
        else:
            result, model = fit_voigt_2d(X, Y, A)

        ## plot the result 
        if show_plot:
            x_fine = np.linspace(x[0], x[-1], 1000)
            y_fine = np.linspace(y[0], y[-1], 1000)
            X_fine, Y_fine = np.meshgrid(x_fine, y_fine, indexing='ij')
            fit_fine = model.func(X_fine, Y_fine, **result.best_values)

            fig, axs = plt.subplots(2, 3, figsize=(15, 10), layout='tight')
            if rotation:
                fig.suptitle('Voigt rotated fit')
            else:   
                fig.suptitle('Voigt fit')
            vmax = np.nanpercentile(A, 95)

            ax = axs[0, 0]
            art = ax.pcolor(img, vmin=0, vmax=vmax, shading='auto')
            # print(result.params['centerx'].value)
            ax.add_patch(Circle((result.params['centery'].value, result.params['centerx'].value), (x[-1]-x[0])/2, fill = False, edgecolor = 'r', lw = 1))
            plt.colorbar(art, ax=ax, label='z')
            ax.set_title('Data')

            ax = axs[0, 1]
            art = ax.pcolor(X, Y, A, vmin=0, vmax=vmax, shading='auto')
            plt.colorbar(art, ax=ax, label='z')
            ax.set_title('Data with ROI')

            ax = axs[0, 2]
            ax_twin = ax.twiny()
            # ax.plot(x, np.sum(A, axis=1)/np.max(np.sum(A, axis=0)), 'o-', c='b', label='along x raw')
            # ax.plot(x_fine, np.sum(fit_fine, axis=1)/np.max(np.sum(fit_fine, axis=0)), '--', c='b', label='along x fit')
            # ax_twin.plot(y, np.sum(A, axis=0)/np.max(np.sum(A, axis=0)),'o-', c='r', label='along y raw')
            # ax_twin.plot(y_fine, np.sum(fit_fine, axis=0)/np.max(np.sum(fit_fine, axis=0)), '--', c='r', label='along y fit')
            ax.plot(x, np.sum(A, axis=1)/len(y), 'o-', c='b', label='along x raw')
            ax.plot(x_fine, np.sum(fit_fine, axis=1)/len(y_fine), '--', c='b', label='along x fit')
            ax_twin.plot(y, np.sum(A, axis=0)/len(x),'o-', c='r', label='along y raw')
            ax_twin.plot(y_fine, np.sum(fit_fine, axis=0)/len(x_fine), '--', c='r', label='along y fit')
            ax.set_ylabel('count / ROI size')
            ax_twin.set_xlabel('y (px)')
            ax_twin.legend(loc='upper left')
            ax.legend(loc='upper right')
            ax.set_title('Binned profile')

            ax = axs[1, 0]
            fit = model.func(X, Y, **result.best_values)
            art = ax.pcolor(X, Y, fit, vmin=0, vmax=vmax, shading='auto')
            plt.colorbar(art, ax=ax, label='z')
            ax.set_title('Fit')

            ax = axs[1, 1]
            fit = model.func(X, Y, **result.best_values)
            art = ax.pcolor(X, Y, A-fit, vmin=-vmax, vmax=vmax, shading='auto')
            plt.colorbar(art, ax=ax, label='z')
            ax.set_title('Data - Fit')

            ax = axs[1, 2]
            ax_twin = ax.twiny()
            ax_twin.plot(y, np.sum(A, axis=0)-np.sum(fit, axis=0),'+-', c='r',label='along y')
            ax_twin.set_xlabel('y (px)')
            ax.plot(x, np.sum(A, axis=1)-np.sum(fit, axis=1), '+-', c= 'b', label='along x')
            ax.set_ylabel('residual (data-fit)')
            ax_twin.legend(loc='upper left')
            ax.legend(loc='upper right')
            ax.set_title('binning of the difference')

            for ax in axs[:,1].ravel():
                ax.set_ylabel('x')
                ax.set_xlabel('y')
            for ax in axs[:,0].ravel():
                ax.set_ylabel('x')
                ax.set_xlabel('y')
            for ax in axs[:,2].ravel():
                ax.set_xlabel('x (px)')
            plt.show()
    ## result.params is composed by differents keys 
    ## print(result.params.keys())
    ## example: get center along x : print(result.params['centerx'].value)
    return result.params

def ROI_around(bp, s_roi):
    bp = np.round(bp).astype(np.int16)
    return [bp[1]-s_roi, bp[1]+s_roi, bp[0]-s_roi, bp[0]+s_roi]

def fit_2dvoigt_rotated(img, ROI=None, show_plot=False):
    return fit_2dvoigt(img, ROI, show_plot, True)

def fit_2dgauss(img, ROI=None, show_plot=False):
    if len(np.shape(img))>2:
        x = np.arange(np.size(img, axis=1)) 
        y = np.arange(np.size(img, axis=2)) 
        X, Y = np.meshgrid(x, y, indexing='ij')
    else:
        if ROI!=None:
            A = make_ROI(img,*ROI)
            x = np.arange(np.size(A, axis=0))+ROI[2] 
            y = np.arange(np.size(A, axis=1))+ROI[0] 
        else: 
            A = img
            x = np.arange(np.size(A, axis=0)) 
            y = np.arange(np.size(A, axis=1)) 
        ## create axis of ROI
        ## create meshgrid
        X, Y = np.meshgrid(x, y, indexing='ij')

        ## reshape for the model
        x_ = np.reshape(X, (np.shape(X)[0]*np.shape(X)[1], ))
        y_ = np.reshape(Y, (np.shape(Y)[0]*np.shape(Y)[1], ))
        A_ = np.reshape(A, (np.shape(A)[0]*np.shape(A)[1], ))
        ## create model 
        model = lmfit.models.Gaussian2dModel()
        # A_ = A_ -np.min(A_)
        ## guess of the model 
        params = model.guess(A_, x_, y_)
        error=np.sqrt(1+A_)

        ## create 2D fit 
        result = model.fit(A_, x=x_, y=y_, params=params, weights=1/error)
        # lmfit.report_fit(result)

        ## plot the result 
        if show_plot:
            x_fine = np.linspace(x[0], x[-1], 1000)
            y_fine = np.linspace(y[0], y[-1], 1000)
            X_fine, Y_fine = np.meshgrid(x_fine, y_fine, indexing='ij')
            fit_fine = model.func(X_fine, Y_fine, **result.best_values)

            fig, axs = plt.subplots(2, 3, figsize=(15, 10), layout='tight')
            fig.suptitle('Gaussian fit')
            vmax = np.nanpercentile(A, 95)

            ax = axs[0, 0]
            art = ax.pcolor(img, vmin=0, vmax=vmax, shading='auto')
            plt.colorbar(art, ax=ax, label='z')
            ax.set_title('Data')

            ax = axs[0, 1]
            art = ax.pcolor(X, Y, A, vmin=0, vmax=vmax, shading='auto')
            plt.colorbar(art, ax=ax, label='z')
            ax.set_title('Data with ROI')

            ax = axs[0, 2]
            ax_twin = ax.twiny()
            ax.plot(x, np.sum(A, axis=1)/np.max(np.sum(A, axis=0)), 'o-', c='b', label='along x raw')
            ax.plot(x_fine, np.sum(fit_fine, axis=1)/np.max(np.sum(fit_fine, axis=0)), '--', c='b', label='along x fit')
            ax_twin.plot(y, np.sum(A, axis=0)/np.max(np.sum(A, axis=0)),'o-', c='r', label='along y raw')
            ax_twin.plot(y_fine, np.sum(fit_fine, axis=0)/np.max(np.sum(fit_fine, axis=0)), '--', c='r', label='along y fit')
            ax.set_ylabel('count normalized')
            ax_twin.set_xlabel('y (px)')
            ax_twin.legend()
            ax.legend()
            ax.set_title('Binned profile')

            ax = axs[1, 0]
            fit = model.func(X, Y, **result.best_values)
            art = ax.pcolor(X, Y, fit, vmin=0, vmax=vmax, shading='auto')
            plt.colorbar(art, ax=ax, label='z')
            ax.set_title('Fit')

            ax = axs[1, 1]
            fit = model.func(X, Y, **result.best_values)
            art = ax.pcolor(X, Y, A-fit, vmin=-vmax, vmax=vmax, shading='auto')
            plt.colorbar(art, ax=ax, label='z')
            ax.set_title('Data - Fit')

            ax = axs[1, 2]
            ax_twin = ax.twiny()
            ax_twin.plot(y, np.sum(A-fit, axis=0),'+-', c='r',label='along y')
            ax_twin.set_xlabel('y (px)')
            ax.plot(x, np.sum(A-fit, axis=1), '+-', c= 'b', label='along x')
            ax.set_ylabel('count difference')
            ax.legend()
            ax_twin.legend()
            ax.set_title('binning of the difference')

            for ax in axs[:,1].ravel():
                ax.set_ylabel('x')
                ax.set_xlabel('y')
            for ax in axs[:,0].ravel():
                ax.set_ylabel('x')
                ax.set_xlabel('y')
            for ax in axs[:,2].ravel():
                ax.set_xlabel('x (px)')
            plt.show()
    ## result.params is composed by differents keys 
    ## print(result.params.keys())
    ## example: get center along x : print(result.params['centerx'].value)
    return result.params


def lorentzian2d(x, y, amplitude=1., centerx=0., centery=0., sigmax=1., sigmay=1.,
                 rotation=0):
    """Return a two dimensional lorentzian.

    The maximum of the peak occurs at ``centerx`` and ``centery``
    with widths ``sigmax`` and ``sigmay`` in the x and y directions
    respectively. The peak can be rotated by choosing the value of ``rotation``
    in radians.
    """
    xp = (x - centerx)*np.cos(rotation) - (y - centery)*np.sin(rotation)
    yp = (x - centerx)*np.sin(rotation) + (y - centery)*np.cos(rotation)
    R = (xp/sigmax)**2 + (yp/sigmay)**2

    return 2*amplitude*lorentzian(R)/(np.pi*sigmax*sigmay)


def fit_2dlorentzian(img, ROI=None, show_plot=False):
    if len(np.shape(img))>2:
        x = np.arange(np.size(img, axis=1)) 
        y = np.arange(np.size(img, axis=2)) 
        X, Y = np.meshgrid(x, y, indexing='ij')
    else:
        if ROI!=None:
            A = make_ROI(img,*ROI)
            x = np.arange(np.size(A, axis=0))+ROI[2] 
            y = np.arange(np.size(A, axis=1))+ROI[0] 
        else: 
            A = img
            x = np.arange(np.size(A, axis=0)) 
            y = np.arange(np.size(A, axis=1)) 
        ## create axis of ROI
        ## create meshgrid
        X, Y = np.meshgrid(x, y, indexing='ij')

        ## reshape for the model
        x_ = np.reshape(X, (np.shape(X)[0]*np.shape(X)[1], ))
        y_ = np.reshape(Y, (np.shape(Y)[0]*np.shape(Y)[1], ))
        A_ = np.reshape(A, (np.shape(A)[0]*np.shape(A)[1], ))
        ## create model 
        model = lmfit.Model(lorentzian2d, independent_vars=['x', 'y'])

        ## initial parameters
        params = model.make_params(amplitude=10, centerx=x_[np.argmax(A_)],
                           centery=y_[np.argmax(A_)])
        params['rotation'].set(value=.1, min=0, max=np.pi/2)
        params['sigmax'].set(value=1, min=0)
        params['sigmay'].set(value=2, min=0)
        error=np.sqrt(1+A_)

        ## build the model 
        result = model.fit(A_, x=x_, y=y_, params=params, weights=1/error)

        ## plot the result 
        if show_plot:
            x_fine = np.linspace(x[0], x[-1], 1000)
            y_fine = np.linspace(y[0], y[-1], 1000)
            X_fine, Y_fine = np.meshgrid(x_fine, y_fine, indexing='ij')
            fit_fine = model.func(X_fine, Y_fine, **result.best_values)

            fig, axs = plt.subplots(2, 3, figsize=(15, 10), layout='tight')
            fig.suptitle('Lorentzian fit')
            vmax = np.nanpercentile(A, 95)

            ax = axs[0, 0]
            art = ax.pcolor(img, vmin=0, vmax=vmax, shading='auto')
            plt.colorbar(art, ax=ax, label='z')
            ax.set_title('Data')
            ax.set_ylabel('x')
            ax.set_xlabel('y')


            ax = axs[0, 1]
            art = ax.pcolor(X, Y, A, vmin=0, vmax=vmax, shading='auto')
            plt.colorbar(art, ax=ax, label='z')
            ax.set_title('Data with ROI')

            ax = axs[0, 2]
            ax_twin = ax.twiny()
            ax.plot(x, np.sum(A, axis=1)/np.max(np.sum(A, axis=0)), 'o-', c='b', label='along x raw')
            ax.plot(x_fine, np.sum(fit_fine, axis=1)/np.max(np.sum(fit_fine, axis=0)), '--', c='b', label='along x fit')
            ax_twin.plot(y, np.sum(A, axis=0)/np.max(np.sum(A, axis=0)),'o-', c='r', label='along y raw')
            ax_twin.plot(y_fine, np.sum(fit_fine, axis=0)/np.max(np.sum(fit_fine, axis=0)), '--', c='r', label='along y fit')
            ax.set_ylabel('count normalized')
            ax_twin.set_xlabel('y (px)')
            ax_twin.legend(loc='upper left')
            ax.legend(loc='upper right')
            ax.set_title('Binned profile')

            ax = axs[1, 0]
            fit = model.func(X, Y, **result.best_values)
            art = ax.pcolor(X, Y, fit, vmin=0, vmax=vmax, shading='auto')
            plt.colorbar(art, ax=ax, label='z')
            ax.set_title('Fit')

            ax = axs[1, 1]
            fit = model.func(X, Y, **result.best_values)
            art = ax.pcolor(X, Y, A-fit, vmin=-vmax, vmax=vmax, shading='auto')
            plt.colorbar(art, ax=ax, label='z')
            ax.set_title('Data - Fit')

            ax = axs[1, 2]
            ax_twin = ax.twiny()
            ax_twin.plot(y, np.sum(A-fit, axis=0),'+-', c='r',label='along y')
            ax_twin.set_xlabel('y (px)')
            ax.plot(x, np.sum(A-fit, axis=1), '+-', c= 'b', label='along x')
            ax.set_ylabel('count difference')
            ax.legend(loc='upper right')
            ax_twin.legend(loc='upper left')
            ax.set_title('binning of the difference')

            for ax in [axs[0,1], axs[1,1], axs[1,0]]:
                ax.set_ylabel('y')
                ax.set_xlabel('x')
            for ax in axs[:,2].ravel():
                ax.set_xlabel('x (px)')
            plt.show()
    ## result.params is composed by differents keys 
    ## print(result.params.keys())
    ## example: get center along x : print(result.params['centerx'].value)
    return result.params



def get_pos_2Dgaussian_fit(img, ROI, show_plot=False):
    result = fit_2dgauss(img, ROI, show_plot)
    return np.array([result['centerx'].value, result['centery'].value])

def get_pos_2Dlorentzian_fit(img, ROI, show_plot=False):
    result = fit_2dlorentzian(img, ROI, show_plot)
    return np.array([result['centerx'].value, result['centery'].value])

def get_pos_2Dvoigt_fit(img, ROI, show_plot=False):
    result = fit_2dvoigt(img, ROI, show_plot)
    return np.array([result['centerx'].value, result['centery'].value])

def get_pos_2Dvoigt_rotated_fit(img, ROI, show_plot=False):
    result = fit_2dvoigt_rotated(img, ROI, show_plot)
    return np.array([result['centerx'].value, result['centery'].value])

def get_pos_pseudovoigt_fit(img, ROI, show_plot=False):
    result = fit_PseudoVoigt(img,ROI, show_plot)
    return np.array([result['centerx'], result['centery']])


def get_pos_around(img, bp, s_roi, method='2D_voigt_rotated', show_plot=False):
    bp = np.round(bp).astype(np.int16)
    ROI = [bp[1]-s_roi, bp[1]+s_roi, bp[0]-s_roi, bp[0]+s_roi]
    return get_pos(img, ROI, method, show_plot)

def get_pos(img, ROI, method='2D_voigt_rotated', show_plot=False):
    if method == '2D_gaussian':
        return get_pos_2Dgaussian_fit(img, ROI, show_plot)
    elif method == '2D_lorentzian':
        return get_pos_2Dlorentzian_fit(img, ROI, show_plot)
    elif method == 'pseudo-voigt':
        return get_pos_pseudovoigt_fit(img, ROI, show_plot)
    elif method == '2D_voigt':
        return get_pos_2Dvoigt_fit(img, ROI, show_plot)
    elif method == '2D_voigt_rotated':
        return get_pos_2Dvoigt_rotated_fit(img, ROI, show_plot)
    else:
        print('argument not valable')
        return False

def pseudovoigt(x,I,mu,gamma,eta,offset,base):
    sigma = gamma/(2*np.sqrt(2*np.log(2)))
    G = 1/sigma/np.sqrt(2*np.pi)*np.exp(-(x-mu)**2/(2*sigma**2)) #Gaussian
    L = 1/np.pi*gamma/2/((x-mu)**2+(gamma/2)**2) #Lorentzian
    return I*(eta*G+(1-eta)*L)+offset+base*x

def momentsPV(data, shift=None):   #accepts 1D arrays
    offset = data.min()
    I = data.max()-offset
    base = (data[-1]-data[0])/len(data)
    tmp = np.where(data>(offset+I/2))
    try:
        gamma = tmp[0][-1]-tmp[0][0]
    except:
        gamma = 0
    if shift==None:
        mu = np.argmax(data)
    else: 
        mu = np.argmax(data) + shift
    eta = 1
    return I,mu,gamma,eta,offset,base
    
def FitPseudoVoigt(data,params, x=0):
    if len(x) == 0:
        x = np.arange(len(data))
    bounds = ([params[0]/1.2,x[0],params[2]/300.,0,-np.inf,-np.inf],[params[0]*300,x[-1],len(data),1,np.inf,np.inf])
    popt, pcov = curve_fit(pseudovoigt, x, data, params, bounds = bounds)
    return popt
    
def make_ROI(img, x0, x1,y0,y1):
    out = img[y0:y1,x0:x1]
    return out
    

def center_of_mass(arr):
    """
    Calculate the center of mass of a 2D numpy array.
    
    Parameters:
    arr (numpy.ndarray): 2D array representing a mass distribution.
    
    Returns:
    tuple: (x_center, y_center) representing the center of mass.
    """
    # Get the indices of the array
    y, x = np.indices(arr.shape)
    
    # Calculate total mass
    total_mass = np.sum(arr)
    
    # Calculate the center of mass
    x_center = np.sum(x * arr) / total_mass
    y_center = np.sum(y * arr) / total_mass
    
    return x_center, y_center

def get_center_mass(lists_img, ROI):
    c_mass=[]
    for img in lists_img:
        img_reduced = make_ROI(img,*ROI)
        x, y = center_of_mass(img_reduced)
        c_mass.append(np.array([ROI[2] + y, ROI[0]+x]))
    return c_mass



#####################################################################
#Pseudovoigt_fit
def fullfitPV(list_imgs,ROI):
    FHB_list =[]
    FVB_list =[]
    for img in list_imgs:
        peak = make_ROI(img,*ROI)
        FHB = np.sum(peak, axis=1) #horizontal binning
        FHB_list.append(FHB)
        FVB = np.sum(peak, axis=0) #vertical binning
        FVB_list.append(FVB)
    params1d_H_list = []
    i=0
    for spectra in FHB_list:
        # We get the right initial value with momentsPV
        moments = momentsPV(spectra)
        try:
            # we fit the data with FitPseudoVoigt(raw data, intial data)
            params = FitPseudoVoigt(spectra,moments)
        except:
            # If we get an error during the fitting we give it PV fit with zero intensity
            print("error H",i)
            params=[0,0,1,1,0,0]
        params1d_H_list.append(params)        
        i=i+1
    params1d_V_list = []
    i=0
    for spectra in FVB_list:
        moments = momentsPV(spectra, ROI[2])
        try:
            params = FitPseudoVoigt(spectra,moments)
        except:
            print("error V",i)
            params=[0,0,1,1,0,0]
            
        params1d_V_list.append(params)        
        i=i+1
        
    params1d_H_arr = np.array(params1d_H_list)
    params1d_V_arr = np.array(params1d_V_list)
    h_H = [0 for i in range(len(list_imgs))]
    h_V = [0 for i in range(len(list_imgs))]
    for i in range(len(list_imgs)):
        # The height is the intensity at the mean position 
        h_H[i] = pseudovoigt(params1d_H_arr[i,1], *params1d_H_arr[i])
        h_V[i] = pseudovoigt(params1d_V_arr[i,1], *params1d_V_arr[i])
    h_H = np.array(h_H)
    h_V = np.array(h_V)

    FWHM_H = params1d_H_arr[:,2]/(2*np.sqrt(2*np.log(2)))
    FWHM_V = params1d_V_arr[:,2]/(2*np.sqrt(2*np.log(2)))

    return params1d_H_arr, params1d_V_arr, FHB_list, FVB_list, h_H, h_V, FWHM_H, FWHM_V
    

#Pseudovoigt_fit
def fit_PseudoVoigt(img,ROI, show_plot=False):
    A = make_ROI(img,*ROI)
    if ROI!=None:
        A = make_ROI(img,*ROI)
        x = np.arange(np.size(A, axis=0))+ROI[2] 
        y = np.arange(np.size(A, axis=1))+ROI[0] 
    else: 
        A = img
        x = np.arange(np.size(A, axis=0)) 
        y = np.arange(np.size(A, axis=1)) 
    spectra_x = np.sum(A, axis=1) #horizontal binning
    spectra_y = np.sum(A, axis=0) #vertical binning
    result = {}

    moments_x = momentsPV(spectra_x, ROI[2])

    moments_y = momentsPV(spectra_y, ROI[0])
    try:
        # we fit the data with FitPseudoVoigt(raw data, intial data)
        params_x = FitPseudoVoigt(spectra_x,moments_x, x)
    except:
        # If we get an error during the fitting we give it PV fit with zero intensity
        print("error fitting along x")
        params=[0,0,1,1,0,0]

    try: 
        params_y =  FitPseudoVoigt(spectra_y,moments_y, y)
    except:
        # If we get an error during the fitting we give it PV fit with zero intensity
        print("error fitting along y")
        params=[0,0,1,1,0,0]   

    result['heightx'] = pseudovoigt(params_x[1], *params_x)
    result['heighty'] = pseudovoigt(params_y[1], *params_y)
    result['centerx'] = params_x[1]
    result['centery'] = params_y[1]
    result['Ix'] = params_x[0]
    result['Iy'] = params_y[0]
    result['gammax'] = params_x[2]
    result['gammay'] = params_y[2]
    result['etax'] = params_x[3]
    result['etay'] = params_y[3]
    result['offsetx'] = params_x[4]
    result['offsety'] = params_y[4]
    result['basex'] = params_x[5]
    result['basey'] = params_y[5]
    result['FWHMx'] = params_x[2]/(2*np.sqrt(2*np.log(2)))
    result['FWHMy'] = params_y[2]/(2*np.sqrt(2*np.log(2)))

    if show_plot:
        x_fine = np.linspace(x[0], x[-1], 1000)
        y_fine = np.linspace(y[0], y[-1], 1000)
        X, Y = np.meshgrid(x, y, indexing='ij')
        # X_fine, Y_fine = np.meshgrid(x_fine, y_fine, indexing='ij')
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle('Pseudo-Voigt fit')
        vmax = np.nanpercentile(A, 95)

        ax = axs[0, 0]
        art = ax.pcolor(img, vmin=0, vmax=vmax, shading='auto')
        plt.colorbar(art, ax=ax, label='z')
        ax.set_title('Data')
        ax.set_ylabel('x')
        ax.set_xlabel('y')

        ax = axs[0, 1]
        art = ax.pcolor(X, Y, A, vmin=0, vmax=vmax, shading='auto')
        plt.colorbar(art, ax=ax, label='z')
        ax.set_title('Data with ROI')
        ax.set_xlabel('x')
        ax.set_ylabel('y')


        ax = axs[1, 0]
        ax.set_title('binning along x')
        high1= 1.1*np.max(spectra_x)
        ax.scatter(x, spectra_x, label='data ')
        ax.plot(x_fine, pseudovoigt(x_fine, *params_x), label= 'fit')
        ax.set_xlabel(r'$x$ (pixel)')
        ax.set_ylabel(r'$I$ (count)')
        ax.set_ylim(0, high1)
        # draw the mean position
        ax.axvline(x = result['centerx'], ymin=0, ymax=result['heightx']/high1, linewidth = 3, color='r', label='center=%.1f px'%(result['centerx']))
        # draw the max intensity  
        ax.axhline(y = result['heightx'], xmin=0, xmax=0.95, linewidth=3, color='b', linestyle='--', label='height=%.1e'%(result['heightx']))
        ax.axhline(y = result['offsetx'], xmin=0, xmax=0.95, linewidth=3, color='y', linestyle='--', label='offset=%.1e'%(result['offsetx']))

        ax.axhline(y = (result['heightx']/2+result['offsetx']), xmin=(result['centerx']-result['FWHMx']-x[0])/(x[-1]-x[0]), xmax=(result['centerx']+result['FWHMx']-x[0])/(x[-1]-x[0]), linewidth=3, color='g', linestyle='-', label= 'FWHM=%.1f px'%(result['FWHMx']))
        ax.grid()
        ax.legend(loc='upper left', fontsize=9)

        ax = axs[1,1]
        ax.set_title('binning along y')
        high2 = 1.1*np.max(spectra_y)
        ax.scatter(y, spectra_y, label='data ')
        ax.plot(y_fine, pseudovoigt(y_fine, *params_y), label= 'fit')
        ax.set_xlabel(r'$y$ (pixel)')
        ax.set_ylabel(r'$I$ (count)')
        ax.set_ylim(0, high2)
        # draw the mean position
        ax.axvline(x = result['centery'], ymin=0, ymax=result['heighty']/high2, linewidth = 3, color='r', label='center=%.1f px'%(result['centery']))
        # draw the max intensity  
        ax.axhline(y = result['heighty'], xmin=0, xmax=0.95, linewidth=3, color='b', linestyle='--', label='height=%.1e'%(result['heighty']))
        ax.axhline(y = result['offsety'], xmin=0, xmax=0.95, linewidth=3, color='y', linestyle='--', label='offset=%.1e'%(result['offsety']))
        # draw the FHMW
        # ax.axhline(y = (result['heightx']+result['offsetx'])/2., xmin=(result['centerx']-result['FWHMx'])/(x[-1]-x[0]), xmax=(result['centerx']+result['FWHMx'])/(x[-1]-x[0]), linewidth=3, color='g', linestyle='-')
        ax.axhline(y = (result['heighty']/2+result['offsety']), xmin=(result['centery']-result['FWHMy']-y[0])/(y[-1]-y[0]), xmax=(result['centery']+result['FWHMy']-y[0])/(y[-1]-y[0]), linewidth=3, color='g', linestyle='-', label='FWHM=%.1f px'%(result['FWHMy']))
        ax.grid()
        ax.legend(loc='upper left', fontsize=9)
        plt.show()
        
    return result

def line_select_callback(eclick, erelease):
    global ROI
    'eclick and erelease are the press and release events'
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    ROI = [np.int(x1),np.int(x2),np.int(y1),np.int(y2)]


def toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)
        
        
        
def get_ROI(image, max_c=3e3):
    #image = Image.open(filename)
    
    fig, current_ax = plt.subplots()
    #image = np.flipud(ndimage.rotate(image, 90))
    plt.imshow(image,cmap='jet', vmin = -1, vmax =max_c)
    toggle_selector.RS = RectangleSelector(current_ax, line_select_callback,
                                       drawtype='box', useblit=True,
                                       button=[1, 3],  # don't use middle button
                                       minspanx=5, minspany=5,
                                       spancoords='pixels',
                                       interactive=True)
    plt.connect('key_press_event', toggle_selector)
    plt.show()
    print('ok')
    frame = ROI 
    return ROI
    
    
    
def get_all(image, frame, save_fit=None): #gives everything on the fit and print the fit 
# take juste ONE image !!! 
# return Ih, Iv, hH, hV, posH, posV, s_h, s_v, count 
# positions are in the full camera frame (adjusted with the ROI)
    global ROI
    ROI = frame   
    #fit 
    p1dH,p1dV,FHB_l,FVB_l,h_H,h_V, s_h, s_v = fullfitPV([image],ROI)
    peak = make_ROI(image,*ROI)
    count = np.sum(peak)
    if save_fit != None:
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        FHB = np.sum(peak, axis=1) #horizontal summing along x axis so variable along y
        high1= 1.1*np.max(FHB)
        x = np.arange(np.size(FHB))
        axs[0].scatter(x, FHB)
        axs[0].plot(x, pseudovoigt(x, *p1dH[0]))
        axs[0].set_xlabel(r'$y$ [pixel]')
        axs[0].set_ylabel(r'$I$ [count]')
        axs[0].set_ylim(0, high1)
        # draw the mean position
        axs[0].axvline(x = p1dH[0, 1], ymin=0, ymax=h_H[0]/high1, linewidth = 3, color='r')
        # draw the max intensity  
        axs[0].axhline(y = h_H[0], xmin=0, xmax=0.95, linewidth=3, color='b', linestyle='--')
        axs[0].axhline(y = p1dH[0,4], xmin=0, xmax=0.95, linewidth=3, color='y', linestyle='--')
        # draw the FHMW
        axs[0].axhline(y = (h_H[0]+p1dH[0,4])/2., xmin=(p1dH[0,1]-s_h)/(x[-1]+1), xmax=(p1dH[0, 1]+s_h)/x[-1], linewidth=3, color='g', linestyle='-')
        axs[0].grid()
        #See fit along x
        FVB = np.sum(peak, axis=0) #vertical summing along y axis so variable along x
        high2 = 1.1*np.max(FVB)
        x = np.arange(np.size(FVB))
        axs[1].scatter(x, FVB)
        axs[1].plot(x, pseudovoigt(x, *p1dV[0]))
        axs[1].set_xlabel(r'$x$ [pixel]')
        axs[1].set_ylabel(r'$I$ [count]')
        axs[1].axvline(x = p1dV[0, 1], ymin=0, ymax=h_V[0]/high2, linewidth = 3, color='r')
        axs[1].axhline(y = h_V[0], xmin=0, xmax=0.95, linewidth=3, color='b', linestyle='--')
        axs[1].axhline(y = p1dV[0,4], xmin=0, xmax=0.95, linewidth=3, color='y', linestyle='--')
        axs[1].axhline(y = (h_V[0]+p1dV[0,4])/2., xmin=(p1dV[0,1]-s_v)/(x[-1]+1), xmax=(p1dV[0, 1]+s_v)/x[-1], linewidth=3, color='g', linestyle='-')
        axs[1].set_ylim(0, high2)
        axs[1].grid()
        fig.savefig(save_fit)
        plt.close()
        plt.cla()
        plt.clf()

    
              
    I_h = p1dH[:,0]
    I_v = p1dV[:,0]   
    return I_h, I_v, h_H, h_V, ROI[2] + p1dH[:,1],ROI[0]+p1dV[:,1], s_h, s_v, count, h_H - p1dH[:,4], h_V - p1dV[:,4]
    
    
    
# def get_pos(image, frame): #take nothing return coordinate of the peak with voigtfit
#     global ROI
#     ROI = frame 
            
#     #fit 
#     p1dH,p1dV,FHB_l,FVB_l,h_H,h_V, s_b, s_v = fullfitPV(image,ROI)   
#     #print(p1dH[0])
#     #print(p1dV[0])
#     return ROI[0]+p1dV[:,1], ROI[3] - p1dH[:,1]
 
 
    
def get_intensity(image, frame): #image is a np array of image
    global ROI
    ROI = frame 
            
    p1dH,p1dV,FHB_l,FVB_l,h_H,h_V, s_b, s_v = fullfitPV(image,ROI)
    #p1dH : parameters horizontal fit return array with : x,I,mu,gamma,eta,offset,base 

    
    I_h = p1dH[:,0]
    I_v = p1dV[:,0]
    return I_h, I_v, h_H, h_V
    
def get_height_no_offset(image, frame):
    global ROI
    ROI = frame     
    p1dH,p1dV,FHB_l,FVB_l,h_H,h_V, s_b, s_v = fullfitPV([image],ROI)    
    h_H = h_H - p1dH[:,4]
    h_V = h_V - p1dV[:,4]
    return h_H, h_V 
    
    
def get_count(image, frame):
    global ROI
    ROI = frame
    count = []
    for img in image: 
        peak = make_ROI(img,*ROI)
        count.append(np.mean(peak))
    return np.array(count)
    
    
    