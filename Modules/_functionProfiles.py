import vide as vu
import numpy as np

from vide import periodic_kdtree as pkd
import multiprocessing as mp
from . import _cosmology as cosmo

def CalculateDistances(data, index):
    num_inputs = np.size(data['inputs'])
    print("\r\tVoids done: %d/%d (%3.2f %%)"%(index+1, num_inputs, 100*(index+1)/num_inputs),end='')
    void_center = data['void_center'][index]
    void_radius = data['void_radii'][index]

    indices = data['tree'].query_ball_point(void_center, r= data['rmax']*void_radius)
    tracers_positions = data['tree'].data[indices]

    #Calculate Weights
    weights = data['tracers_z'][indices]
    weights = np.interp(weights, data['mean_density_z'] ,data['mean_density'])
    weights = 1./weights

    number_tracers = np.shape(tracers_positions)[0]
    distances = np.zeros(number_tracers)

    # Calculate distances
    for i, part in enumerate(tracers_positions):
        distances[i] = np.sqrt(np.sum((part-void_center)**2.))
    
    dist_array = np.array([index, distances, weights], dtype=object)

    return dist_array

def CalculatePureDistances(data, index):
    num_inputs = np.size(data['inputs'])
    
    print("\r\tVoids done: %d/%d (%3.2f %%)"%(index+1, num_inputs, 100*(index+1)/num_inputs),end='')
    void_center = data['void_center'][index]
    void_radius = data['void_radii'][index]
    try:
        indices = data['tree'].query_ball_point(void_center, r= data['rmax']*void_radius)
    except:
        print(void_radius, void_center, index)
        raise

    tracers_positions = data['tree'].data[indices]

    #Calculate Weights
    #weights = data['tracers_z'][indices]
    #weights = np.interp(weights, data['mean_density_z'] ,data['mean_density'])
    #weights = 1./weights

    number_tracers = np.shape(tracers_positions)[0]
    distances = np.zeros(number_tracers)

    # Calculate distances
    for i, part in enumerate(tracers_positions):
        distances[i] = np.sqrt(np.sum((part-void_center)**2.))
    weights = np.ones(np.shape(distances))
    dist_array = np.array([index, distances, weights], dtype=object)

    return dist_array

def IntersectingSpheresVolume(R, r, d):
    numerator = np.pi*(R+r-d)**2.*(d**2.+2.*d*r-3.*r**2.+2.*d*R+6.*r*R-3.*R**2.)
    denominator = 12.*d
    return numerator/denominator


def JackKnifeResampling(init_profile, distances, weights, rbins):

    n = np.shape(distances)[0]
    profiles = np.zeros((n, np.size(rbins[:-1])))
    for i in np.arange(n):
        print('\r\tJackKnife: %d/%d'%(i+1,n), end='')
        profile, profile_bins = np.histogram(distances[i], bins=rbins, weights=weights[i])
        profiles[i] = init_profile-profile
    
    print('\tDone!')
    
    profiles = profiles/(n-1)
    mean_profile = np.sum(profiles,axis=0)/n 
    variance = (n-1)/n*np.sum(np.power(np.subtract(profiles,mean_profile),2.), axis=0)
    return mean_profile, variance

def JackKnifeResampling_Profiles(profiles, bins):
    n = np.shape(profiles)[0]
    temp_profiles = np.zeros((n, np.size(bins[:-1])))
    init_profile = np.sum(profiles, axis=0)
    for i in np.arange(n):
        print('\r\tJackKnife Profiles: %d/%d'%(i+1,n), end='')
        temp_profiles[i] = (init_profile - profiles[i])

    temp_profiles= temp_profiles/(n-1)

    mean_profile = np.sum(temp_profiles, axis=0)/n
    variance = (n-1)/n*np.sum(np.power(np.subtract(temp_profiles,mean_profile),2.),axis=0)
    print(' Done!')
    return mean_profile, variance


def jackknife(profiles):
    n0, n1 = np.shape(profiles)
    if n0 <= 1:
        raise ValueError('jackknife: data must contain at least 2 array')
    
    resamples = np.empty([n0, n1])
    for i in range(n0):
        resamples[i] = np.mean(np.delete(profiles, i, axis=0), axis=0)
    
    return resamples

def jackknife_results(profiles):

    resamples = jackknife(profiles)
    mean = np.mean(resamples, axis=0)

    n = np.shape(resamples)[0]
    variance = (n-1)/n*np.sum(np.power(np.subtract(resamples, mean),2.), axis=0)
    std = np.sqrt(variance)

    return resamples, mean, std
