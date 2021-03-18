from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


from ._colors import *
from ._Profiles import Profiles
from ._Fitting import HSW_Fitting

class CVA(HSW_Fitting):

    def __init__(self, InputFile, **kwargs):

        """
        Class for carrying out a first Cosmic Voids Analysis (CVA).
        The analysis can be carried out for several cosmology at the same time.

        Parameters:
        -----------
            
            InputFile : str
                Path to the catalog directory

        Other parameters:
        -----------------

            **kwargs

                Omega_M_array : 1-D float array

                bins : 1-D array

                Catalog_Omega_M : float

                fast : boolean

                new : boolean
        """
        
        print(f"{col.HEADER2}**************************************************{col.END}")
        print(f"{col.HEADER2}Initializing Comic Voids Analysis{col.END}")
        self.InputFile = InputFile
        self.CreateDirectories(self.InputFile)
        
        self.Omega_M_array = kwargs.get('Omega_M_array', None)
        if np.any(self.Omega_M_array == None):
            raise InputError('No Omega_M_array inserted')
        
        self.ranges = kwargs.get('ranges', None)
        if np.any(self.ranges == None):
            self.ranges = np.arange(40,161,20)
            print(f'{col.NOTICE}Standard voids radii ranges used!{col.END}')
        
        self.bins = kwargs.get('bins', None)
        if np.any(self.bins == None):
            self.bins = np.append(0,np.cumsum(np.flip(np.diff(np.logspace(0,np.log10(4),41)-1))))
            print(f'{col.NOTICE}Standard bins values used!{col.END}')

        if kwargs.get('Omega_M',None) != None:
            raise InputError("Don't use Omega_M as kwargs but Omega_M_array")
        print(f"{col.HEADER2}**************************************************{col.END}")


    def profiles_analysis(self, **kwargs):
        
        print(f"{col.HEADER2}**************************************************{col.END}")
        print(f"{col.HEADER2}Analysis:{col.END}")
        self.compare_same = kwargs.get('compare_same',False)
        estimator=kwargs.get('estimator', '')

        n_Om = np.size(self.Omega_M_array)
        n_r = np.size(self.ranges[:-1])
        n_b = np.size(self.bins)

        # Initialize profiles_tot
        self.profiles_tot = np.zeros((n_Om*n_r,n_b-1))
        self.profiles_errors_tot = np.zeros((n_Om*n_r,n_b-1))
        self.profiles_bins_tot = np.zeros((n_Om*n_r,n_b))

        self.studied_ranges = '_R'+str(int(n_r))+'_'+str(int(self.ranges[0]))+'_'+str(int(self.ranges[1]-self.ranges[0]))

        for i, Om in enumerate(self.Omega_M_array):

            self.add_M = '_M' + str(Om).split('.')[-1]

            print(f"{col.NOTICE}--------------------------------------------------{col.END}")
            print(f"{col.NOTICE}Omega_M: {Om}{col.END}")
            if self.compare_same:
                text = r"\Omega_M"
                print(f"{col.NOTICE}Binning voids' radii for {text} = 0.270{col.END}")
                nameFile_Profiles = self.folder_profiles+'/Profile_sameR'+estimator+self.studied_ranges+self.add_M+'.h5'
            else:
                nameFile_Profiles = self.folder_profiles+'/Profile'+estimator+self.studied_ranges+self.add_M+'.h5'
            
            # Retrieve profiles or calculate it
            new_computation = True
            if Path(nameFile_Profiles).is_file():
                data = self.Upload(nameFile_Profiles)
                if np.all(data['profiles']['bins'] == self.bins):
                    
                    print(f'{col.NOTICE}Retrieving Profiles: {col.END}', nameFile_Profiles)
                    catalog, voids, tracers = self.LoadCatalog(self.InputFile, Omega_M = Om, **kwargs)
                    self.profiles = data['profiles']['profiles']
                    self.profiles_errors = data['profiles']['profiles_errors']
                    self.profiles_bins = data['profiles']['profiles_bins']
                    new_computation = False
            new_computation = True
            if new_computation:
                catalog, voids, tracers = self.LoadCatalog(self.InputFile, Omega_M = Om, **kwargs)
                # fsky key in MeanDensity to consider masked part
                self.MeanDensity()
                self.CreateRandoms()
                #self.DistanceVoidsTracers()
                #self.StackVoids(self.ranges, bins=self.bins, compare_same= self.compare_same)
                #self.randoms = self.tracers
                self.CorrelationVoidsTracersRandoms()
                data_temp = {
                            'bins':self.bins,
                            'profiles':self.profiles,
                            'profiles_errors':self.profiles_errors,
                            'profiles_bins':self.profiles_bins
                            }
                self.Save(nameFile_Profiles, profiles=data_temp)

            ii=0
            for j in np.arange(i*n_r, i*n_r+n_r, 1):
                self.profiles_tot[j] = self.profiles[ii]
                self.profiles_errors_tot[j] = self.profiles_errors[ii]
                self.profiles_bins_tot[j] = self.profiles_bins[ii]
                ii += 1

            print(f"{col.NOTICE}--------------------------------------------------{col.END}")
        print(f"{col.HEADER2}**************************************************{col.END}")
    

    def profiles_plot(self, xmax=3.0, estimator=''):

        print(f"{col.HEADER2}**************************************************{col.END}")
        print(f"{col.HEADER2}Plotting:{col.END}")
        if self.compare_same:
            nameFile_ProfilesPlot = self.folder_results+'/PlotProfiles_sameR'+estimator+self.studied_ranges+self.add_M+'.pdf'
        else:
            nameFile_ProfilesPlot = self.folder_results+'/PlotProfiles'+estimator+self.studied_ranges+self.add_M+'.pdf'

        n_r = np.size(self.ranges[:-1])
        n_Om = np.size(self.Omega_M_array)
        styles = ['.','x','v', '^', '+', 'p']
        lin_styles = ['-', '-.',':','--']

        # Cycling over Omegas and Ranges
        for i, Om in enumerate(self.Omega_M_array):
            for j in np.arange(n_r):
                k = i*n_r+j
                profile = self.profiles_tot[k]
                profile_errors = self.profiles_errors_tot[k]
                x = (self.profiles_bins_tot[k][1:]+self.profiles_bins_tot[k][:-1])/2.
                label = 'Radii range: ['+str(self.ranges[j])+','+str(self.ranges[j+1])+') Mpc/h'

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
                plt.plot(x, profile, linewidth=0.2, linestyle=lin_style, color=colo)
                plt.errorbar(x, profile, yerr=profile_errors, fmt=style, markersize=2,
                                color= colo, ecolor=colo, elinewidth=0.2, label=label)
        
        plt.grid()
        plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', prop={'size':8})
        plt.xlabel(r'$R/R_v$')
        plt.ylabel(r'$d(x)/d_{mean}-1$')
    
        plt.xlim(0.0, xmax)
        print('Saving on '+self.print_folder(nameFile_ProfilesPlot, last=1, start='')+' ...')
        plt.savefig(nameFile_ProfilesPlot, format='pdf', bbox_inches='tight')
        print('Done!')
        print(f"{col.HEADER2}**************************************************{col.END}")


    def profiles_fitting(self,**kwargs):
        
        ## Kwargs:
        # General:
        #        method : fit, MCMC, both
        #        new : default False
        #        fitting_limits : Radii range in which to fit
        #    MCMC:
        #        n_walkers : int (default 64)
        #        n_iteration : int (default 5000)

        method = kwargs.get('method', '')

        print(f"{col.HEADER2}**************************************************{col.END}")
        print(f"{col.HEADER2}Fitting {method}:{col.END}")
        if method == 'fit':
            HSW_Fitting.fitting(self, **kwargs)
            HSW_Fitting.plot(self, **kwargs)

        elif method == 'MCMC':
            HSW_Fitting.fitting_MCMC(self, **kwargs)
            HSW_Fitting.plot(self, **kwargs)

        elif method == 'both':
            HSW_Fitting.fitting_MCMC(self, **kwargs)
            print(self.parameters)
            print(self.parameters_errors)
            self.parameters_MCMC = self.parameters
            HSW_Fitting.fitting(self, **kwargs)
            print(self.parameters)
            print(self.parameters_errors)
        else:
            print(f'{col.FAIL}No method =\"{method}\" found{col.END}')
            raise InputError("Wrong fitting method inputted. Chose between: fit, MCMC")

        print(f"{col.HEADER2}**************************************************{col.END}")

