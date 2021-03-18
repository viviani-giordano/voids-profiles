import errno, os, sys, time
from timeit import default_timer as timer
from pathlib import Path

import vide as vu
import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl
import multiprocessing as mp
from vide import periodic_kdtree as pkd
import scipy
from scipy.optimize import curve_fit
import emcee


from ._colors import *
from ._Load import Load
from ._Profiles import Profiles
from .Modules import _functionFitting as func


def HSW(r, rs, alpha, Rv, beta, deltac):
    """
    HSW (Hamaus-Sutter-Wendelt) function for the universal void density profile
    See: Hamaus et al. (2014)
    """
    numerator = 1-(r/rs)**alpha
    denominator = 1+(r/Rv)**beta

    return deltac*numerator/denominator 


def HSW_MCMC(param , param_limits, r, profile ,profile_err):

    rs, alpha, Rv, beta, deltac = param
    rs_lim, alpha_lim, Rv_lim, beta_lim, deltac_lim = param_limits
    numerator = 1-(r/rs)**alpha
    denominator = 1+(r/Rv)**beta
    model = deltac*numerator/denominator 
    
    res = func.log_probability(param, param_limits, model, profile, profile_err)
    return res


def HSW_offset(r, rs, alpha, Rv, beta, deltac, offset):
    """
    HSW (Hamaus-Sutter-Wendelt) function for the universal void density profile
    See: Hamaus et al. (2014)
    """
    numerator = 1-(r/rs)**alpha
    denominator = 1+(r/Rv)**beta

    return deltac*numerator/denominator +offset


def HSW_MCMC_offset(param , param_limits, r, profile ,profile_err):

    rs, alpha, Rv, beta, deltac, offset = param
    rs_lim, alpha_lim, Rv_lim, beta_lim, deltac_lim, offset_lim = param_limits
    numerator = 1-(r/rs)**alpha
    denominator = 1+(r/Rv)**beta
    model = deltac*numerator/denominator +offset
    
    res = func.log_probability(param, param_limits, model, profile, profile_err)
    return res


class HSW_Fitting(Profiles):

    def fitting_MCMC(self, **kwargs):

        # Kwargs
        fitting_limits = kwargs.get('fitting_limits',None)
        n_walkers = kwargs.get('n_walkers', 64)
        n_iteration = kwargs.get('n_iteration', 5000)
        new = kwargs.get('new', False)
        offset = kwargs.get('offset', False)
        self.add_offset = ''
        if offset:
            self.add_offset = '_offset'
        if np.any(fitting_limits != None):
            print(f'\t{col.NOTICE}Fitting data between {fitting_limits[0]}  and {fitting_limits[1]} R/R_v{col.END}')
        print(f'{col.NOTICE}n_walkers : {n_walkers}{col.END}')
        print(f'{col.NOTICE}n_iteration : {n_iteration}{col.END}')

        # File name
        if self.compare_same:
            nameFile_FittingHSW_MCMC = self.folder_profiles+'/FittingHSW_MCMC_sameR'+self.add_offset+self.studied_ranges+self.add_M+'.h5'
        else:
            nameFile_FittingHSW_MCMC = self.folder_profiles+'/FittingHSW_MCMC'+self.add_offset+self.studied_ranges+self.add_M+'.h5'

        # Check if fit has been already calculated
        if (Path(nameFile_FittingHSW_MCMC).is_file() and not(new)):
            data = self.Upload(nameFile_FittingHSW_MCMC)
            if np.all(data['MCMC']['Omega_M_array']== self.Omega_M_array):
                print(f'{col.NOTICE}Retrieving parameters:{col.END}')
                self.parameters = data['MCMC']['parameters']
                self.parameters_errors = data['MCMC']['parameters_errors']
                return
        
        print(f'{col.NOTICE}Calculating parameters:{col.END}')
        n_r = np.size(self.ranges[:-1])
        n_Om = np.size(self.Omega_M_array)  

        # Initialize parameters vectors
        # Parameters:
        # rs, alpha, Rv, beta, deltac, offset, log_f 
        n_par = 5
        if offset:
            n_par = 6
        parameters = np.zeros((np.shape(self.profiles_tot)[0], n_par))
        parameters_errors = np.zeros((np.shape(self.profiles_tot)[0], n_par))
        
        # Cycling over Omegas and Ranges
        for i, Om in enumerate(self.Omega_M_array):
            for j in np.arange(n_r):
                k = i*n_r+j
                profile = self.profiles_tot[k]
                profile_errors = self.profiles_errors_tot[k]
                x = (self.profiles_bins_tot[k][1:]+self.profiles_bins_tot[k][:-1])/2.

                np.random.seed(42)

                # Study values inside the fitting limits
                if np.any(fitting_limits != None):
                    n_profile = np.size(profile)
                    l_vec = np.ones(n_profile)
                    for l in range(np.size(profile)):
                        if fitting_limits[0]<=x[l]<=fitting_limits[1]:
                            l_vec[l] = 0
                    l_vec = np.argwhere(l_vec) 
                    profile = np.delete(profile, l_vec)
                    profile_errors = np.delete(profile_errors, l_vec)
                    x = np.delete(x, l_vec)

                # Run the MCMC algorithm                
                if offset:
                    # Define parameters bounds
                    lower_bound = np.array([0.5, 0, 0.8, 0, -2,-0.1])
                    upper_bound = np.array([1.1, 5, 1.2, 10, 0, 0.1 ])
                    
                    initial_guess = np.array([0.8, 2, 1., 1, -0.8,0])
                    random_start = initial_guess + np.random.rand(n_walkers, n_par)*(upper_bound-lower_bound)/10.
                    
                    bounds = np.zeros((n_par,2))
                    for l in range(n_par):
                        bounds[l] = [lower_bound[l], upper_bound[l]]
                    # Parameters for HSW_MCMC:
                    # param_limits, r, profile ,profile_err 
                    sampler = emcee.EnsembleSampler(n_walkers, n_par, HSW_MCMC_offset, args=(bounds, x, profile, profile_errors))
                    sampler.run_mcmc(random_start, n_iteration, progress=True)
                
                else:
                    lower_bound = np.array([0.5, 0, 0.8, 0, -2])
                    upper_bound = np.array([1.1, 5, 1.2, 10, 0])
                    
                    initial_guess = np.array([0.8, 2, 1., 1, -0.8])
                    random_start = initial_guess + np.random.rand(n_walkers, n_par)*(upper_bound-lower_bound)/10.
                    
                    bounds = np.zeros((n_par,2))
                    for l in range(n_par):
                        bounds[l] = [lower_bound[l], upper_bound[l]]
                    # Parameters for HSW_MCMC:
                    # param_limits, r, profile ,profile_err 
                    sampler = emcee.EnsembleSampler(n_walkers, n_par, HSW_MCMC, args=(bounds, x, profile, profile_errors))
                    sampler.run_mcmc(random_start, n_iteration, progress=True)

                flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
                parameters_temp = np.zeros(n_par)
                parameters_err_temp = np.zeros(n_par)

                # Get the percentiles: 16, 50, 84
                for l in range(n_par):
                    mcmc = np.percentile(flat_samples[:,l], [16, 50,84])
                    mcmc_err = np.mean(np.diff(mcmc))

                    parameters_temp[l] = mcmc[1]
                    parameters_err_temp[l] = mcmc_err

                # get most probable values
                prob_vec = sampler.get_log_prob(discard=100, thin=15, flat=True)
                prob_idx = np.argmax(prob_vec)
                parameters[k] = flat_samples[prob_idx,:] 
                self.flat_samples = flat_samples
                #parameters[k] = parameters_temp
                parameters_errors[k] = parameters_err_temp

        # Saving the results of the fitting
        data_temp ={
                    'parameters':parameters,
                    'parameters_errors': parameters_errors,
                    'Omega_M_array':self.Omega_M_array,
                    'ranges': self.ranges
                    }
        self.Save(nameFile_FittingHSW_MCMC, MCMC=data_temp)

        self.parameters = parameters
        self.parameters_errors = parameters_errors
        

    def fitting(self, **kwargs):

        # Kwargs
        fitting_limits = kwargs.get('fitting_limits',None)
        new = kwargs.get('new', False)
        offset = kwargs.get('offset', False)
        self.add_offset = ''
        if offset:
            self.add_offset = '_offset'
        if np.any(fitting_limits != None):
            print(f'\t{col.NOTICE}Fitting data between {fitting_limits[0]}  and {fitting_limits[1]} R/R_v{col.END}')

        # File name
        if self.compare_same:
            nameFile_FittingHSW = self.folder_profiles+'/FittingHSW_sameR'+self.add_offset+self.studied_ranges+self.add_M+'.h5'
        else:
            nameFile_FittingHSW = self.folder_profiles+'/FittingHSW'+self.add_offset+self.studied_ranges+self.add_M+'.h5'
        
        # Check if fit has been already calculated
        if (Path(nameFile_FittingHSW).is_file() and not(new)):
            data = self.Upload(nameFile_FittingHSW)
            if np.all(data['fit']['Omega_M_array']== self.Omega_M_array):
                print(f'{col.NOTICE}Retrieving parameters:{col.END}')
                self.parameters = data['fit']['parameters']
                self.parameters_errors = data['fit']['parameters_errors']
                return

        print(f'{col.NOTICE}Calculating parameters:{col.END}')
        n_r = np.size(self.ranges[:-1])
        n_Om = np.size(self.Omega_M_array)  

        # Initialize parameters vectors
        n_par = 5
        if offset:
            n_par=6
        parameters = np.zeros((np.shape(self.profiles_tot)[0],n_par))
        parameters_errors = np.zeros((np.shape(self.profiles_tot)[0],n_par))     
        
        # Cycling over Omegas and Ranges
        for i, Om in enumerate(self.Omega_M_array):
            for j in np.arange(n_r):
                k = i*n_r+j
                profile = self.profiles_tot[k]
                profile_errors = self.profiles_errors_tot[k]
                x = (self.profiles_bins_tot[k][1:]+self.profiles_bins_tot[k][:-1])/2.
                
                # Study values inside the fitting limits
                if np.any(fitting_limits != None):
                    n_profile = np.size(profile)
                    l_vec = np.ones(n_profile)
                    for l in range(np.size(profile)):
                        if fitting_limits[0]<=x[l]<=fitting_limits[1]:
                            l_vec[l] = 0
                    l_vec = np.argwhere(l_vec) 
                    profile = np.delete(profile, l_vec)
                    profile_errors = np.delete(profile_errors, l_vec)
                    x = np.delete(x, l_vec)

                # Fit the curve
                if offset:
                    popt, pcov = curve_fit( HSW_offset,
                                            x,
                                            profile,
                                            sigma=profile_errors,
                                            p0=[0.8,2,1., 1,-0.8, 0],
                                            bounds = ([0.5, 0, 0.8, 0, -2, -0.1], [1.1, 10, 1.2, 10, 0, 0.1]),
                                            maxfev=1000000
                                            )
                else:
                    popt, pcov = curve_fit( HSW,
                                            x,
                                            profile,
                                            sigma=profile_errors,
                                            p0=[0.8,2,1., 1, -0.8],
                                            bounds = ([0.5, 0, 0.8, 0, -2], [1.1, 10, 1.2, 10, 0]),
                                            maxfev=1000000
                                            )

                parameters[k] = popt
                parameters_errors[k] = np.diag(pcov)

        # Saving the results of the fitting
        data_temp ={
                    'parameters':parameters,
                    'parameters_errors': parameters_errors,
                    'Omega_M_array':self.Omega_M_array
                    }
        self.Save(nameFile_FittingHSW, fit=data_temp)

        self.parameters = parameters
        self.parameters_errors = parameters_errors


    def plot(self,estimator = '',**kwargs):
        
        # Kwargs
        xmax = kwargs.get('xmax', 3.5)
        method = kwargs.get('method', '')
        offset = kwargs.get('offset', False)
        self.add_offset = ''
        if offset:
            self.add_offset = '_offset'

        # File name
        if self.compare_same:
            nameFile_PlotFittingHSW = self.folder_results+'/PlotFittingHSW'+estimator+'_'+method+'_sameR'+self.add_offset+self.studied_ranges+self.add_M+'.pdf'
        else:
            nameFile_PlotFittingHSW = self.folder_results+'/PlotFittingHSW'+estimator+'_'+method+self.add_offset+self.studied_ranges+self.add_M+'.pdf'

        # Initialize figure and parameters
        n_r = np.size(self.ranges[:-1])
        n_Om = np.size(self.Omega_M_array)
        styles = ['.','x','v', '^', '+', 'p']
        lin_styles = ['-', '-.',':','--']
        fig, ax = plt.subplots(ncols=1, nrows=1)

        # Cycling over Omegas and Ranges
        for i, Om in enumerate(self.Omega_M_array):
            for j in np.arange(n_r):
                k = i*n_r+j
                profile = self.profiles_tot[k]
                profile_errors = self.profiles_errors_tot[k]
                x = (self.profiles_bins_tot[k][1:]+self.profiles_bins_tot[k][:-1])/2.
                label = 'Radii range: ['+str(self.ranges[j])+','+str(self.ranges[j+1])+') Mpc/h'

                x_model = np.linspace(0.0,xmax, num=10000)
                if offset:
                    y_model = HSW_offset(x_model,*self.parameters[k])
                else:
                    y_model = HSW(x_model,*self.parameters[k])

                # Distinguish if only one Omega_m is analyzed
                if n_Om >1:
                    label2 = r'$\Omega_M$: '+str(Om)
                    label = label +'\n'+label2
                    style = styles[j]
                    lin_style = lin_styles[j]
                    shades = col.colors(n_Om)
                    colo = shades[i]
                else:
                    style = '.'
                    lin_style = '-'
                    shades2 = col.colors(n_r)
                    colo = shades2[j]

                ax.errorbar(x, profile, yerr=profile_errors,fmt=style, linewidth=0., markersize=2,
                                color= colo, ecolor=colo, elinewidth=0.2, label=label)
                ax.plot(x_model, y_model, linewidth=0.2, linestyle='-', color=colo)
                #plt.fill_between(x,y_min,y_max, alpha =0.2, facecolor=colo)
        

        ax.legend(bbox_to_anchor=(1.05,1), loc='upper left', prop={'size':8})
        ax.set_xlabel(r'$R/R_v$')
        ax.set_ylabel(r'$\delta (x)/\delta_{mean}-1$')
    
        ax.grid()
        ax.set_xlim(0.0, xmax)
        #ax.set_ylim(-0.85, 0.42)
        print('Saving on '+self.print_folder(nameFile_PlotFittingHSW, last=1,start='')+' ...')
        fig.savefig(nameFile_PlotFittingHSW, format='pdf', bbox_inches='tight')
        print('Done!')

