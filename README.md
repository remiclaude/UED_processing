# UED_processing
In this folder are all the scripted used to process and plot the result shown in the publication ...


- find_peak.py 
Every functions related to fit of peaks. 

- function_analysis.py
Every python functions useful for data analysis (e.g. remove_hot_pixel, shift_0th_order, etc)

- treat_pickle.ipynb
Jupyter notebook using find_peak.py and function_analysis.py. It takes the pickle file from RAW_SORTED folder (available in 10.5281/zenodo.14760926) and do the following: removes hot pixels; shifts the images to keep the 0th order (unscattered beam) at a fix position; computes the calibration constant from the distance between the peaks; average the images along the symmetry axis. 

- UED_data_paper.ipynb
Jupyter notebook for plotting the panels of figure 2 in the manuscript. The panels were merged and labels were added on illustrator

- UED_SI_fig.ipynb 
Jupyter notebook for plotting the panels of figure S4 (in the supplementay materials). The panels were merged and labels were added on illustrator

- UED_SI_fit.ipynb 
Jupyter notebook for plotting the panels of figure S4 (in the supplementay materials). It uses the fitting method 2D_voigt wrote in find_peak.py

- UED_exp_fig.ipynb
Jupyter notebook for plotting diffraction map and normalized diffraction maps shown in figure 2 of the main manuscript

- qEELS_exp_fig.ipynb
Jupyter notebook for plotting the electron energy loss spectrum shown in figure 2 of the main manuscript

