import errno, os, sys, time
from timeit import default_timer as timer
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from ._colors import *
from .Modules import _cosmology as cosmo
from ._Load import Load


class Statistics(Load):

    def MeanDensity(self, **kwargs):

        print(f'{col.HEADER}**************************************************{col.END}')
        print(f'{col.HEADER}MeanDensity:{col.END}')
        bins = kwargs.get('bins', 100)
        fsky = kwargs.get('fsky', 1.)
        start = timer()
        nameFile_MeanDensity = self.folder_data+'/MeanDensity'+self.add_M+'.h5'

        if Path(nameFile_MeanDensity).is_file():
            print(f'{col.STEP}Data retrieved from:{col.END}')
            print('\t'+self.print_folder(nameFile_MeanDensity))
            data = self.Upload(nameFile_MeanDensity)
            self.nzrn = data['mean_density']['nzrn']
            self.zr_centers = data['mean_density']['zr_centers']
        
        else:

            # 1st STEP: Redshift Range and Distribution
            print(f"{col.STEP}1/4 Tracers' Redshift{col.END}")
            z = self.tracers['redshift']
            N = np.size(z)
            zmin = np.min(z)
            zmax = np.max(z)
            print('\tRedshift range: [%f; %f]'%(zmin, zmax))
            nz, zbins = np.histogram(z, bins=bins)
            z_centers = (zbins[1:]+zbins[:-1])/2.

            # 2nd STEP: Recreate the distribution using the randoms
            print(f"{col.STEP}2/4 Recreate the distribution using randoms{col.END}")
            N_randoms = 5*N
            Randoms = np.random.rand(N_randoms).astype(np.float32) # Random array 
            Randoms = Randoms*(zmax-zmin)+zmin # Random redshifts (Rescaling)
            nr = np.interp(Randoms,z_centers,nz) # Interpolate nz(z) and calculate it for Randoms
            Randoms = np.random.choice(Randoms,N_randoms,p=nr/np.sum(nr)) # Simulate the density distribution as a function of the redshift
            nzr,zrbins = np.histogram(Randoms, bins=bins) 
            zr_centers= (zrbins[1:]+zrbins[:-1])/2.
            print("\tValues Used:")
            print("\tOmega_M: ",self.Omega_M)
            print("\tOmega_L: ",self.Omega_L,"\t(Omega_K = ", 1-(self.Omega_M+self.Omega_L),")")
            print('\tfsky: ', fsky)
            V = fsky*4.*np.pi/3.*((cosmo.AD(zbins[1:],self.Omega_M,self.Omega_L))**3.-(cosmo.AD(zbins[:-1],self.Omega_M,self.Omega_L))**3.)
            Vr = fsky*4.*np.pi/3.*((cosmo.AD(zrbins[1:],self.Omega_M,self.Omega_L))**3.-(cosmo.AD(zrbins[:-1],self.Omega_M,self.Omega_L))**3.)
            nzn = nz/V
            nzrn = nzr*N/N_randoms/Vr

            # 3rd STEP: Creating and Saving Plot
            nameFile_MeanDensityPlot = self.folder_results + '/MeanDensity'+self.add_M+'.pdf'
            nameFile_MeanDensityPlot_log = self.folder_results + '/MeanDensity_log'+self.add_M+'.pdf'
            print(f"{col.STEP}3/4 Creating and saving the MeanDensity plot{col.END}")
            print('\tSaving under: '+ nameFile_MeanDensityPlot)
            plt.plot(z_centers, nzn,color='r', linewidth=1, label='Tracers distribution')
            plt.plot(zr_centers, nzrn,color='b',linestyle='--', linewidth=0.5, label='Random distribution')
            plt.xlabel(r'$z$')
            plt.ylabel(r'$n(z) \ [h^3 Mpc^{-3}]$')
            plt.title('Mean Density Distribution')
            plt.ticklabel_format(axis='both',style='sci', scilimits=(-2,2))
            plt.grid()
            plt.legend()
            plt.savefig(nameFile_MeanDensityPlot,format='pdf', bbox_inches='tight')
            plt.yscale('log')
            plt.savefig(nameFile_MeanDensityPlot_log,format='pdf', bbox_inches='tight')
            plt.clf()
            
            # 4th STEP: Saving Mean Density Data
            print(f"{col.STEP}4/4 Saving Data{col.END}")
            print('\tSaving under: '+ nameFile_MeanDensity)
            self.nzrn = nzrn
            self.zr_centers = zr_centers
            temp_dict = {'nzrn':nzrn, 
                         'zr_centers':zr_centers}
            self.Save(nameFile_MeanDensity, mean_density = temp_dict )
        
        end = timer()
        print(f'MeanDensity RunTime: {end-start}')
        print(f'{col.STEP}Output:{col.END}')
        print('\tnzrn\n\tzr_centers')
        print(f'{col.HEADER}**************************************************{col.END}')
    

    def VoidAbundance(self, **kwargs):
        
        bins = kwargs.get('bins', 100)
        xlim = kwargs.get('xlim', 300.)
        fsky = kwargs.get('fsky', 1.)
        # Get fsky as squaredeg of sphere
        fsky = 4*np.pi* ((180/np.pi)**2) *fsky

        R = self.voids['radius']
        n_voids, r_voids_bins = np.histogram(self.voids['radius'], bins=bins)
        r_voids_centers = (r_voids_bins[1:]+r_voids_bins[:-1])/2.
        dr_voids_centers = (r_voids_centers[1:]-r_voids_centers[:-1]).mean()
        z = self.tracers['redshift']
        zmax, zmin = np.max(z), np.min(z)
        Vol = fsky * (np.pi/180)**2/3  
        Vol = Vol * (cosmo.AD(zmax, self.Omega_M, self.Omega_L)**3 - cosmo.AD(zmin, self.Omega_M, self.Omega_L)**3)

        n_voids = n_voids/Vol/dr_voids_centers*r_voids_centers
        n_voids_err = np.sqrt(n_voids/Vol/dr_voids_centers*r_voids_centers)
        
        self.voidAbundance = n_voids
        self.voidAbundance_err = n_voids_err
        self.voidAbundance_z = r_voids_centers

        nameFile_VoidAbundance = self.folder_results+'/VoidAbundance'+self.add_M+'.png'
        plt.clf()
        fig, ax = plt.subplots(ncols=1, nrows=1)
        ax.plot(r_voids_centers, n_voids)
        ax.errorbar(r_voids_centers, n_voids, yerr=n_voids_err, color='b', fmt='.', ms=6)
        text = r'$N_\nu$ = '+str(len(R))
        plt.figtext(0.65, 0.8, text)
        ax.set_xlabel(r'$R \ [h^{-1}\  Mpc]$')
        ax.set_ylabel(r'$dn(R)/d \ln R\ [h^3\ Mpc^3]$')
        ax.set_yscale('log')
        ax.set_xlim(0., xlim)
        ax.grid()

        plt.savefig(nameFile_VoidAbundance, format='png', bbox_inches='tight')
        plt.clf()


    def VoidDistribution(self, **kwargs):
        R = self.voids['radius']
        z = self.voids['redshift']
        
        nameFile_VoidDistribution = self.folder_results+'/VoidDistribution'+self.add_M+'.png'

        plt.clf()
        fig, ax = plt.subplots(ncols=1, nrows=1)
        ax.scatter(z, R, s=2, c = R, cmap = 'viridis', alpha=0.3)
        ax.set_xlabel(r'$z$')
        ax.set_ylabel(r'$R$')
        ax.grid()
        plt.savefig(nameFile_VoidDistribution, format='png', bbox_inches='tight')


    def TracersDistribution(self, **kwargs):

        _range = kwargs.get('range', [-100,100]+self.voids['boxLen'][:1]/2.)
        _axis = kwargs.get('axis', 'xy')
        _add_axis = '_'+_axis
        x = self.tracers['partPos'][:,0]
        y = self.tracers['partPos'][:,1]
        z = self.tracers['partPos'][:,2]
        redshift = self.tracers['redshift']
        
        # Cutting slice
        if _axis == 'xy':
            idx = np.argwhere(((_range[0]<=z) & (z<=_range[1])))
        elif _axis == 'yz':
            idx = np.argwhere(((_range[0]<=x) & (x<=_range[1])))
        elif _axis == 'zx':    
            idx = np.argwhere(((_range[0]<=y) & (y<=_range[1])))
        else:
            print(f'{col.FAIL}Wrong axis key used. Supported values: xy, yz, zx {coil.END}')
            return
        print('Studied plane: ', _axis)
        print('Number of tracers inside range: ', np.shape(idx))
        x = x[idx]
        y = y[idx]
        z = z[idx]
        redshift = redshift[idx]
        nameFile_TracersDistribution_xy = self.folder_results+'/TracersDistribution'+_add_axis+self.add_M+'.png'
        
        # Creating and Saving Figures
        plt.clf()
        fig, ax = plt.subplots(ncols=1, nrows=1)
        if _axis == 'xy':
            sc = ax.scatter(x, y, s=2, c=redshift, cmap = 'viridis', alpha=0.3)
        elif _axis == 'yz':
            sc = ax.scatter(y, z, s=2, c=redshift, cmap = 'viridis', alpha=0.3)
        else:    
            sc = ax.scatter(z, x, s=2, c=redshift, cmap = 'viridis', alpha=0.3)

        cbar  = fig.colorbar(sc)
        cbar.ax.set_ylabel('redshift')
        ax.set_xlabel(_axis[0])
        ax.set_ylabel(_axis[1])
        ax.set_xlim(0,self.voids['boxLen'][0])
        ax.set_ylim(0,self.voids['boxLen'][1])
        ax.set_aspect(1)
        ax.grid()
        plt.savefig(nameFile_TracersDistribution_xy, format='png', bbox_inches='tight')


    def CreateRandoms(self, **kwargs):

        print(f'{col.HEADER}**************************************************{col.END}')
        print(f'{col.HEADER}CreateRandoms:{col.END}')
        bins = kwargs.get('bins', 100)
        fsky = kwargs.get('fsky', 1.)
        start = timer()
        nameFile_Randoms = self.folder_data+'/Randoms'+self.add_M+'.h5'
        mean_dens = self.nzrn
        mean_dens_z = self.zr_centers

        folder_randoms = '/e/ocean1/nschuster/Giordano/clusters_mock/full/'
        file_randoms = folder_randoms + 'clusters_full_R.fits'

        if Path(file_randoms).is_file():
            print(f'{col.FAIL}NOTE:{col.END} Randoms retrieved from {file_randoms}')
            from astropy.table import Table
            data = Table.read(file_randoms, format='fits')
            randoms_partPos = cosmo.ComovingPosition(data['Z'], data['RA'], data['DEC'], self.Omega_M, self.Omega_L)
            randoms_partPos = randoms_partPos + self.voids['boxLen']/2.

            self.randoms = {'redshift': data['Z'],
                            'partPos': randoms_partPos}
        else:
            if Path(nameFile_Randoms).is_file():
                print(f'{col.STEP}Data retrieved from:{col.END}')
                print('\t'+self.print_folder(nameFile_Randoms))
                data = self.Upload(nameFile_Randoms)
                self.randoms = data['data']
                from mpl_toolkits import mplot3d
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                ax.scatter3D(self.randoms['partPos'][:1000,0],self.randoms['partPos'][:1000,1],self.randoms['partPos'][:1000,2],
                        c= self.randoms['redshift'][:1000], cmap=plt.cm.viridis)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                plt.savefig(self.folder_results+'/Randoms.png', format='png', bbox_inches='tight')
                #self.zr_centers = data['mean_density']['zr_centers']
                Randoms = np.random.rand(1000).astype(np.float32)
                Randoms2 = np.random.rand(1000).astype(np.float32)
                random_points_RA = Randoms*(360)
                random_points_DEC = np.arccos(1-2*Randoms2)*(180/np.pi)-90
                fig = plt.figure()
                plt.scatter(random_points_RA, random_points_DEC)
                plt.xlabel('RA')
                plt.ylabel('DEC')
                plt.savefig(self.folder_results+'/Randoms2D.png', format='png', bbox_inches='tight')
                plt.clf()
    
            else:
                print(f'{col.STEP}Creating the random distribution:{col.END}') 
                factor = 5 # N_randoms = factor*N_tracers
                print(f'{col.STEP}\tN_randoms = {factor} N_tracers{col.END}')
                N_tracers = np.size(self.tracers['redshift'])
                N_randoms = factor*N_tracers
                Randoms = np.random.rand(N_randoms).astype(np.float32)
                
                Randoms2 = np.random.rand(N_randoms).astype(np.float32)
                Randoms3 = np.random.rand(N_randoms).astype(np.float32)
                z = self.tracers['redshift']
                zmin, zmax = np.min(z), np.max(z)
    
                randoms_z = (Randoms*(zmax-zmin))+zmin
                probability  = np.interp(randoms_z, mean_dens_z, mean_dens)
                random_points_z = np.random.choice(randoms_z, N_randoms, p = probability/np.sum(probability))
                random_points_RA = Randoms2*(360)
                random_points_DEC = np.arccos(1-2*Randoms3)*(180/np.pi)-90
                random_partPos_temp = cosmo.ComovingPosition(random_points_z,
                                                              random_points_RA,
                                                              random_points_DEC,
                                                              self.Omega_M,
                                                              self.Omega_L)
                random_partPos = random_partPos_temp + self.voids['boxLen']/2.
    
                self.randoms = {'redshift': random_points_z,
                                'partPos': random_partPos}
    
                # Saving Randoms Data
                print(f"{col.STEP}Saving Data{col.END}")
                print('\tSaving under: '+ nameFile_Randoms)
                self.Save(nameFile_Randoms, data = self.randoms )
        
        end = timer()
        print(f'CreateRandoms RunTime: {end-start}')
        print(f'{col.STEP}Output:{col.END}')
        print('\trandoms')
        print(f'{col.HEADER}**************************************************{col.END}')

