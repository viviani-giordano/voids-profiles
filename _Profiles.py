import errno, os, sys, time
from timeit import default_timer as timer
from pathlib import Path

import vide as vu
import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl
import multiprocessing as mp
from vide import periodic_kdtree as pkd
from scipy.spatial import cKDTree as Tree

from ._colors import *
from .Modules import _cosmology as cosmo
from .Modules import _functionProfiles as func
from ._Load import Load
from ._Statistics import Statistics

class Profiles(Statistics, Load):

    def DistanceVoidsTracers(self, new=False, rmax=3.):
        print(f'{col.HEADER}**************************************************{col.END}')
        print(f'{col.HEADER}DistanceVoidsTracers:{col.END}')
        self.rmax = rmax
        print(f'Maximum distance considered: {self.rmax} R_e')
        start = timer()
        nameFile_Distances = self.folder_data+'/Distances'+self.add_M+'.hkl'
        nameFile_Weights = self.folder_data+'/Weights'+self.add_M+'.hkl'
        
        if (Path(nameFile_Distances).is_file() and Path(nameFile_Weights).is_file() and not(new)):
            print(f'{col.STEP}Data retrieved from:{col.END}')
            print('\t'+self.print_folder(nameFile_Distances))
            print('\t'+self.print_folder(nameFile_Weights))
            self.distances = self.Upload(nameFile_Distances)
            self.distances_weights = self.Upload(nameFile_Weights)

        else:

            # 1st STEP Creation of the particle tree
            print(f'{col.STEP}\t1/3 Creation of the PeriodicCKDTree{col.END}')
            #position_Tree = pkd.PeriodicCKDTree(self.voids['boxLen'], self.tracers['partPos'])
            position_Tree = Tree(self.tracers['partPos'])
            
            # 2nd STEP Calculating Distances
            print(f'{col.STEP}\t2/3 Calculating distances (in parallel){col.END}')
            start_para = timer()
            inputs = np.arange(np.size(self.voids['radius']))

            # Creating Dictionary
            data_dict = {
                         'tree':position_Tree,
                         'tracers_z': self.tracers['redshift'],
                         'mean_density': self.nzrn,
                         'mean_density_z':self.zr_centers,
                         'rmax':self.rmax,
                         'void_center': self.voids['macrocenter'],
                         'void_radii':self.voids['radius'],
                         'inputs':inputs   
                        }

            pool = mp.Pool(mp.cpu_count())
            results = pool.starmap_async(func.CalculateDistances,[(data_dict,index) for index in inputs]).get()
            pool.close()

            results = np.array(results)
            index = results[:,0]
            self.distances = results[:,1]
            self.distances_weights = results[:,2]
            
            end = timer()
            print('Done!')
            print(f'\tDistances Calculation RunTime: {end-start_para}')

            # 3rd STEP Saving
            print(f'{col.STEP}\t3/3 Saving results in hkl files{col.END}')
            hkl.dump(self.distances, nameFile_Distances, mode='w' )
            hkl.dump(self.distances_weights, nameFile_Weights, mode='w' )

        end = timer()
        print(f'\nDistanceVoidsTracers RunTime: {end-start}')
        print(f'{col.STEP}Outputs:{col.END}')
        print('\tdistances\n\tdistances_weights')
        print(f'{col.HEADER}**************************************************{col.END}')
        

    def StackVoids(self,ranges, bins=100, compare_same=False):      
        print(f'{col.HEADER}**************************************************{col.END}')
        print(f'{col.HEADER}StackVoids:{col.END}')
        if compare_same:
            print(f"{col.UNDERLINE}Considering voids based on their catalog's radii!!{col.END}")
        if type(bins) is int:
            _, rbins = np.histogram(0, bins=bins, range=(0., self.rmax))
        else:
            _, rbins = np.histogram(0, bins=bins)

        profiles = np.zeros((np.size(ranges[:-1]), np.size(rbins[:-1])))
        profiles_bins = np.zeros((np.size(ranges[:-1]), np.size(rbins)))
        profiles_errors = np.zeros((np.size(ranges[:-1]), np.size(rbins[:-1])))
        for i, low in enumerate(ranges[:-1]):
            high = ranges[i+1]

            if compare_same:
                radii_27 = self.AdjustRadius(self.voids['radius'],self.voids['redshift'],flip=True) 
                indices = np.argwhere((radii_27>=low)&(radii_27<high)).reshape(-1)
            else:
                indices = np.argwhere((self.voids['radius']>=low)&(self.voids['radius']<high)).reshape(-1)

            # Getting distances and weights for the relative voids
            print(f'{col.STEP}Distances and weights for the relative voids:{col.END}', end='')
            distances_range = self.distances[indices]
            weights_range = self.distances_weights[indices]
            voids_radii_range = self.voids['radius'][indices]
            voids_center_range = self.voids['macrocenter'][indices]
            
            for ii, distance in enumerate(distances_range):
                void_radius = voids_radii_range[ii]
                void_center = voids_center_range[ii]

                rbin = rbins*void_radius
                centered_void_center = void_center-self.voids['boxLen']/2.
                R_centered = np.sqrt(np.sum(centered_void_center**2.))

                # Calculate shell volume
                shell = np.zeros(np.size(rbin[:-1]))
                for j, r in enumerate(rbin[:-1]):
                    if (rbin[j+1]+R_centered)<self.voids['simulation_radius']:
                        shell[j] = (4.*np.pi/3.)*(rbin[j+1]**3.-r**3.)
                    elif (r+R_centered)<self.voids['simulation_radius']:
                        V2 = func.IntersectingSpheresVolume(self.voids['simulation_radius'], rbin[j+1], R_centered)
                        V1 = (4.*np.pi/3.)*(r**3.)
                        shell[j]=V2-V1
                    else:
                        V2 = func.IntersectingSpheresVolume(self.voids['simulation_radius'], rbin[j+1], R_centered)
                        V1 = func.IntersectingSpheresVolume(self.voids['simulation_radius'], r, R_centered)
                        shell[j]=V2-V1
                    
                index_bin = np.digitize(distance,rbin)-1
                #print(index_bin)
                #print(np.shape(weights_range[ii]))
                #print(np.shape(shell[index_bin]))
                weights_range[ii] = weights_range[ii]/shell[index_bin]
                distances_range[ii] = distance/void_radius
           
            print(' Done!')

            # Jack-Knife Resampling Data
            print(f'{col.STEP}Creating the initial profile:{col.END}', end='')
            init_profile, profile_bins = np.histogram(np.concatenate(distances_range), bins=rbins, weights=np.concatenate(weights_range))
            print(' Done!')
            print(f'{col.STEP}JackKnifeResampling:{col.END}')
            profile, variance = func.JackKnifeResampling(init_profile,distances_range,
                                                            weights_range,
                                                            rbins)

            init_profile, profile_bins = np.histogram(np.concatenate(distances_range), bins=rbins, weights=np.concatenate(weights_range))
            profiles[i] = profile -1
            profiles_bins[i] = profile_bins
            profiles_errors[i] = np.sqrt(variance)

        self.profiles = profiles
        self.profiles_bins = profiles_bins
        self.profiles_errors = profiles_errors
        
        print(f'{col.STEP}Outputs{col.END}:')
        print('\tprofiles\n\tprofiles_bins\n\tprofiles_errors')
        print(f'{col.HEADER}**************************************************{col.END}')
    

    def ProfilesVoidCenterCalculation(self, objects, new, nameFile, nameFile_Weights, weights_factor=1.):
        start_prof = timer()
        if (Path(nameFile).is_file() and Path(nameFile_Weights).is_file() and not(new)):
            print(f'{col.STEP}Data retrieved from:{col.END}')
            print('\t'+self.print_folder(nameFile))
            print('\t'+self.print_folder(nameFile_Weights))
            distances = self.Upload(nameFile)
            distances_weights = self.Upload(nameFile_Weights)

        else:
            # Calculating Distances
            print(f'{col.STEP}Calculating Distances:{col.END}')
            # 1st STEP Creation of the particle tree
            print(f'{col.STEP}\t1/3 Creation of the PeriodicCKDTree{col.END}')
            #position_Tree = pkd.PeriodicCKDTree(self.voids['boxLen'], objects['partPos'])
            position_Tree = Tree(objects['partPos'])

            # 2nd STEP Calculating Distances
            print(f'{col.STEP}\t2/3 Calculating distances (in parallel){col.END}')
            start_para = timer()
            inputs = np.arange(np.size(self.voids['radius']))

            # Creating Dictionary
            data_dict = {
                         'tree':position_Tree,
                         'tracers_z': objects['redshift'],
                         'mean_density': self.nzrn,
                         'mean_density_z':self.zr_centers,
                         'rmax':self.rmax,
                         'void_center': self.voids['macrocenter'],
                         'void_radii': self.voids['radius'],
                         'inputs':inputs   
                        }

            pool = mp.Pool(mp.cpu_count())
            results = pool.starmap_async(func.CalculatePureDistances,[(data_dict,index) for index in inputs]).get()
            pool.close()

            results = np.array(results)
            index = results[:,0]
            distances = results[:,1]
            distances_weights = results[:,2]/weights_factor
            print('Done!')
            end = timer()
            print(f'\tDistances Calculation RunTime: {end-start_para}')
            # 3rd STEP Saving
            print(f'{col.STEP}\t3/3 Saving results in hkl files{col.END}')
            hkl.dump(distances, nameFile, mode='w' )
            hkl.dump(distances_weights, nameFile_Weights, mode='w' )


        #  Calculating Profiles
        print(f'{col.STEP}Single Profiles:{col.END}')
        #profiles = np.zeros((np.size(self.ranges[:-1]), np.size(self.bins[:-1])))
        #profiles_bins = np.zeros((np.size(self.ranges[:-1]), np.size(self.bins)))
        #profiles_errors = np.zeros((np.size(self.ranges[:-1]), np.size(self.bins[:-1])))
        
        single_profiles_array = np.zeros((np.size(distances), np.size(self.bins[:-1])))
        voids_radii = self.voids['radius']

        for i, distance in enumerate(distances):
            single_profiles_array[i], temp = np.histogram(distance, bins=self.bins*voids_radii[i], weights = distances_weights[i])
        print('Shape of the single profiles array:', np.shape(single_profiles_array))
        '''
        # Cycle over the different ranges
        single_profiles_bins = np.zeros()
        for i, low in enumerate(self.ranges[:-1]):
            high = self.ranges[i+1]
            indices = np.argwhere((self.voids['radius']>=low)&(self.voids['radius']<high)).reshape(-1)

            # Getting distances and weights for the relative voids
            distances_range = distances[indices]
            weights_range = distances_weights[indices]
            voids_radii_range = self.voids['radius'][indices]
            voids_center_range = self.voids['macrocenter'][indices]


            # Cycle over the voids inside the range
            profiles_temp = np.zeros(np.size(indices))
            for ii, distance in enumerate(distances_range):
            
                void_radius = voids_radii_range[ii]
                void_center = voids_center_range[ii]

                rbin = self.bins*void_radius
                #centered_void_center = void_center-self.voids['boxLen']/2.
                #R_centered = np.sqrt(np.sum(centered_void_center**2.))
                
                # Calculate shell volume
                shell = np.zeros(np.size(rbin[:-1]))
                for j, r in enumerate(rbin[:-1]):
                    if (rbin[j+1]+R_centered)<self.voids['simulation_radius']:
                        shell[j] = (4.*np.pi/3.)*(rbin[j+1]**3.-r**3.)
                    elif (r+R_centered)<self.voids['simulation_radius']:
                        V2 = func.IntersectingSpheresVolume(self.voids['simulation_radius'], rbin[j+1], R_centered)
                        V1 = (4.*np.pi/3.)*(r**3.)
                        shell[j]=V2-V1
                    else:
                        V2 = func.IntersectingSpheresVolume(self.voids['simulation_radius'], rbin[j+1], R_centered)
                        V1 = func.IntersectingSpheresVolume(self.voids['simulation_radius'], r, R_centered)
                        shell[j]=V2-V1
                   
                index_bin = np.digitize(distance,rbin)-1
                weights_range[ii] = (weights_range[ii]/shell[index_bin])
               

                distances_range[ii] = distance/void_radius
                profiles_temp[ii], temp = np.histogram(distances_range[ii], bins=self.bins)

            single_profiles_array[i] = profiles_temp
            single_profiles_bins[i] = temp
        '''


            
        '''
            # Jack-Knife Resampling Data
            
            init_profile, profile_bins = np.histogram(np.concatenate(distances_range), bins=self.bins, weights=np.concatenate(weights_range))
            print(f'{col.STEP}\tJackKnifeResampling for R_v = [{low},{high}] Mpc/h{col.END}')
            profile, variance = func.JackKnifeResampling(init_profile,distances_range,
                                                            weights_range,
                                                            self.bins)

            init_profile, profile_bins = np.histogram(np.concatenate(distances_range), bins=self.bins, weights=np.concatenate(weights_range))
            profiles[i] = profile
            profiles_bins[i] = profile_bins
            profiles_errors[i] = np.sqrt(variance)
        '''
        
        end = timer()
        print(f'\tProfiles Calculation RunTime: {end-start_prof}')
        #return profiles, profiles_bins, profiles_errors
        return single_profiles_array

    def DavisPeeblesCorrelation(self, DD, DR):
        print(f'{col.STEP}DavisPeeblesCorrelation:{col.END}')
        profiles = np.zeros((np.size(self.ranges[:-1]), np.size(self.bins[:-1])))
        profiles_bins = np.zeros((np.size(self.ranges[:-1]), np.size(self.bins)))
        profiles_errors = np.zeros((np.size(self.ranges[:-1]), np.size(self.bins[:-1])))

        #ratio = np.divide(DD,DR)
        #DDmean = np.mean(DD, axis=0)
        ratio = np.zeros(np.shape(DD))
        for i in np.arange(np.shape(DD)[0]):
            for j in np.arange(np.shape(DD)[1]):
                if DR[i][j]!=0:
                    ratio[i][j] = DD[i][j]/(DR[i][j])
                else:
                    ratio[i][j] = DD[i][j]
            
        self.ratio = ratio
        print('Ratio shape:', np.shape(ratio))

        for i, low in enumerate(self.ranges[:-1]):
            high = self.ranges[i+1]
            indices = np.argwhere((self.voids['radius']>=low)&(self.voids['radius']<high)).reshape(-1)
            ratio_range = ratio[indices]
            print(f'{col.STEP}\tJackKnifeResampling for R_v = [{low},{high}] Mpc/h{col.END}')

            mean, variance = func.JackKnifeResampling_Profiles(ratio_range, self.bins)

            profiles[i] = mean
            profiles_bins[i] = self.bins
            profiles_errors[i] = np.sqrt(variance)

        return profiles, profiles_bins, profiles_errors


    def CorrelationVoidsTracersRandoms(self, new= False, rmax=3.):
        print(f'{col.HEADER}**************************************************{col.END}')
        print(f'{col.HEADER}DistanceVoidsTracers:{col.END}')
        self.rmax = rmax
        print(f'Maximum distance considered: {self.rmax} R_e')
        start = timer()
        nameFile_DistancesDD = self.folder_data+'/DistancesDD'+self.add_M+'.hkl'
        nameFile_WeightsDD = self.folder_data+'/WeightsDD'+self.add_M+'.hkl'
        nameFile_DistancesDR = self.folder_data+'/DistancesDR'+self.add_M+'.hkl'
        nameFile_WeightsDR = self.folder_data+'/WeightsDR'+self.add_M+'.hkl'
        
             
        resultsDD = self.ProfilesVoidCenterCalculation(self.tracers,
                                                     new,
                                                     nameFile_DistancesDD,
                                                     nameFile_WeightsDD)
        #profilesDD = results[0] 
        #profilesDD_bins = results[1]
        #profilesDD_errors = results[2]

        resultsDR = self.ProfilesVoidCenterCalculation(self.randoms,
                                                      new,
                                                      nameFile_DistancesDR,
                                                      nameFile_WeightsDR,
                                                      weights_factor = 5.)
        #profilesDR = results2[0]
        #profilesDR_bins = results2[1]
        #profilesDR_errors = results2[2]
        self.resultsDR = resultsDR
        self.resultsDD = resultsDD

        profiles, profiles_bins, profiles_errors = self.DavisPeeblesCorrelation(resultsDD, resultsDR)

        '''
        profiles = np.zeros((np.size(self.ranges[:-1]), np.size(self.bins[:-1])))
        profiles_bins = np.zeros((np.size(self.ranges[:-1]), np.size(self.bins)))
        profiles_errors = np.zeros((np.size(self.ranges[:-1]), np.size(self.bins[:-1])))

        for i in np.arange(np.size(self.ranges[:-1])):
            n_bins = np.size(profilesDD[i])
            profile_temp = np.zeros(n_bins)
            profile_errors_temp = np.zeros(n_bins)

            for j in np.arange(n_bins):
                profile_temp[j] = profilesDD[i][j]/profilesDR[i][j]-1
                #relative_errorDD = profilesDD_errors[i][j]/profilesDD[i][j]
                #relative_errorDR = profilesDR_errors[i][j]/profilesDR[i][j]
                #profile_errors_temp[j] = profile_temp[j]*np.sqrt((relative_errorDR)**2+(relative_errorDD)**2)
                profile_errors_temp[j] = profilesDD_errors[i][j]

            profiles[i] = profile_temp
            profiles_errors[i] = profile_errors_temp
            profiles_bins[i] = profilesDD_bins[i]
        '''
        self.profiles = profiles-1
        self.profiles_bins = profiles_bins
        self.profiles_errors = profiles_errors
        
        end = timer()
        print(f'\nCorrelationVoidsTracersRandoms: {end-start}')
        print(f'{col.STEP}Outputs:{col.END}')
        print('\tprofiles\n\tprofiles_bins\n\tprofiles_errors')
        print(f'{col.HEADER}**************************************************{col.END}')
