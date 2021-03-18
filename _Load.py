import vide as vu
import numpy as np
import hickle as hkl
import h5py
import errno, os, sys, time
from timeit import default_timer as timer
from pathlib import Path
from ._colors import *
from .Modules import _cosmology as cosmo

class Load:

    def LoadCatalog(self, InputFile,**kwargs):
        
        print(f"{col.HEADER}**************************************************{col.END}")
        print(f"{col.HEADER}LoadCatalog:{col.END}")
        self.Catalog_Omega_M = kwargs.get('Catalog_Omega_M', 0.270)
        print(f'{col.STEP}Catalog_Omega_M used:{col.END} {self.Catalog_Omega_M}')
        self.Omega_M = kwargs.get('Omega_M', 0.307115)
        self.Omega_L = kwargs.get('Omega_L', 1. - self.Omega_M)
        self.add_M = '_M' + str(self.Omega_M).split('.')[-1]

        fast = kwargs.get('fast', False)
        new = kwargs.get('new', False)

        start = timer()
        self.CreateDirectories(InputFile)
        nameFile_VoidsTracers = self.folder_data+'/DataVoidsTracers'+self.add_M+'.h5'
        nameFile_Catalog = self.folder_data+'/Catalog.hkl'
        
        # Upload or Create data files
        if (Path(nameFile_VoidsTracers).is_file() and Path(nameFile_Catalog).is_file() and not(new)):
            print(f"{col.STEP}Data Uploaded from:{col.END}")

            if fast:
                self.catalog = None
            else:    
                print('\t'+self.print_folder(nameFile_Catalog))
                self.catalog = self.Upload(nameFile_Catalog)
            
            print('\t'+self.print_folder(nameFile_VoidsTracers))
            data = self.Upload(nameFile_VoidsTracers)
            self.voids = data['voids']
            self.tracers = data['tracers']
        
        else:
            print(f'{col.STEP}Catalog upload from:{col.END}')
            self.catalog = vu.loadVoidCatalog(InputFile, dataPortion='central',
                                                untrimmed=True, loadParticles=True)

            # Retrieve Data
            print(f"{col.STEP}Calculating the Comoving Positions...{col.END}")
            print(f"{col.STEP}Voids centers...{col.END}", end='')
            voidID = vu.getArray(self.catalog.voids,'voidID')
            R = vu.getArray(self.catalog.voids,'radius')
            Z = vu.getArray(self.catalog.voids,'redshift')
            RA = vu.getArray(self.catalog.voids,'RA')
            Dec = vu.getArray(self.catalog.voids, 'Dec')
            macrocenter_temp = cosmo.ComovingPosition(Z, RA, Dec, self.Omega_M, self.Omega_L)
            macrocenter = macrocenter_temp + self.catalog.boxLen/2.

            R = self.AdjustRadius(R, Z)


            print('Done!')

            print(f"{col.STEP}Tracers...{col.END}", end='')
            numPart = self.catalog.numPartTot
            RApart = vu.getArray(self.catalog.part,'ra')
            Decpart = vu.getArray(self.catalog.part,'dec')
            Zpart = vu.getArray(self.catalog.part,'redshift')/cosmo.c

            partPos_temp =  cosmo.ComovingPosition(Zpart, RApart, Decpart,
                                                self.Omega_M, self.Omega_L)
            partPos = partPos_temp + self.catalog.boxLen/2.
            
            # Estimate simulation radius
            x_diam = np.max(partPos[:,0]) - np.min(partPos[:,0])
            y_diam = np.max(partPos[:,1]) - np.min(partPos[:,1])
            z_diam = np.max(partPos[:,2]) - np.min(partPos[:,2])
            simulation_radius = np.max(np.array([x_diam, y_diam, z_diam]))/2.

            print('Done!')

            # Creating dictionaries
            self.voids = {
                'numVoids':self.catalog.numVoids,
                'boxLen':self.catalog.boxLen,
                'ranges':self.catalog.ranges,
                'simulation_radius': simulation_radius,
                'voidID': voidID,
                'macrocenter':macrocenter,
                'macrocenter_x':macrocenter[:,0],
                'macrocenter_y':macrocenter[:,1],
                'macrocenter_z':macrocenter[:,2],
                'radius':R,
                'redshift':Z,
                'RA':RA,
                'Dec':Dec
                }

            self.tracers = {
                'numPart':numPart,
                'partPos':partPos,
                'x':partPos[:,0],
                'y':partPos[:,1],
                'z':partPos[:,2],
                'ra':RApart,
                'dec':Decpart,
                'redshift':Zpart
                }

            if not(Path(nameFile_Catalog).is_file()):
                print(f"{col.STEP}Saving catalog on the .hkl file{col.END}")
                hkl.dump(self.catalog, nameFile_Catalog, mode='w')
            
            self.Save(nameFile_VoidsTracers, voids = self.voids, tracers = self.tracers)

        end = timer()
        print(f"LoadCatalog RunTime: {end-start}")
        print(f"{col.HEADER}**************************************************{col.END}")

        return self.catalog, self.voids, self.tracers


    def Upload(self, nameFile):
        type_file = str(nameFile).split('.')[-1]
    
        if type_file == 'hkl':
            data = hkl.load(nameFile)
        elif type_file == 'h5':
            f = h5py.File(nameFile, 'r+')

            data = {}
            for key in list(f.keys()):
                temp_dict = {}
                for val in list(f[key].keys()):
                    temp_dict[val] = np.array(f[key].get(val))
                data[key] = temp_dict 
        else:
            sys.exit(f"""{col.FAIL}TypeError:{col.END} Error while running Upload.\n File has
                    a wrong type. File types accepted: .hkl, .h5""")
        return data


    def Save(self, nameFile, **kwargs):

        hf = h5py.File(nameFile,'w')
        for key, value in kwargs.items():
            temp_gr = hf.create_group(key)
            for key_dict in value:
                temp_gr.create_dataset(key_dict, data= value[key_dict])
        hf.close()


    def CreateDirectories(self, InputFile):
        self.path = os.getcwd()
        if InputFile[-1]=='/':
            folder = InputFile[:-1]
        else:
            folder = InputFile
        
        folder = folder.split('/')[-1]
        self.folder = '/'+folder +'_analysis'
        self.folder_data = self.path + self.folder + '/Data'
        self.folder_profiles = self.path + self.folder + '/Data/Profiles'
        self.folder_results = self.path + self.folder + '/Results'

        try:
            os.makedirs(self.folder_data)
            os.makedirs(self.folder_profiles)
            os.makedirs(self.folder_results)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    

    def AdjustRadius(self, R, Z, flip=False):

        H_027 = 100*np.sqrt(self.Catalog_Omega_M*(1.+Z)**3+(1.-self.Catalog_Omega_M))
        DA_027 = cosmo.AD(Z,self.Catalog_Omega_M, 1.-self.Catalog_Omega_M)
        H = 100*np.sqrt(self.Omega_M*(1.+Z)**3+self.Omega_L)
        DA = cosmo.AD(Z, self.Omega_M, self.Omega_L)
        q1 = H_027/H
        q2 = DA/DA_027

        if flip:
            return q1**(-1/3)*q2**(-2/3)*R
        else:
            return q1**(1/3)*q2**(2/3)*R


    def print_folder(self, string, last=3, split='/', start='.'):
        last = -last
        vec = string.split(split)[last:]
        string = start
        for folder in vec:
            string += '/'+folder
        return string    

