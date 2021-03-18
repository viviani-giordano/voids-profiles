import numpy as np
from scipy import integrate
import multiprocessing as mp




c = 299792.458 #km/s
G = 6.67385e-11 #N m^2 kg^-2

def InvHubble(z, Omega_M, Omega_L):
    """
    Calculate the inverse Hubble rate c/H(z) in units of 1/h
    """
    return c/(100*np.sqrt(Omega_M*(1+z)**3+Omega_L))
InvHubble = np.vectorize(InvHubble)

def AD(z, Omega_M, Omega_L):
    """
    Calculate the diameter distance for a FLAT Universe (given
    value for Omega Matter and Lambda) given a certain redshift z.
    """
    return integrate.quad(InvHubble, 0.,z, args=(Omega_M,Omega_L))[0]
AD = np.vectorize(AD)

def parallelize_AD(z, Omega_M, Omega_L):
    """
    Version of AD() that can be used by the multiprocessing library. 
    """
    return integrate.quad(InvHubble, 0.,z, args=(Omega_M,Omega_L))[0]


def ComovingPosition(redshift, RA, Dec, Omega_M, Omega_L):

    """
    Calculates the comoving positions of an array of 3-D points given the redshift and
    the angular displacement (RA and Dec) of the latter.
    Notice that RA and Dec must be provided in degrees.

    Parameters:
    ----------
        redshift : 1-D array
            Array containing the redshift of the points

        RA : 1-D array
            Right ascension values of the points. The values must be expressed in degrees
            and be contained in the interval: [0,360)

        Dec : 1-D array
            Declination values of the points. The values must be expressed in degrees and
            be contained in the interval: [-90,90]
            
        Omega_M : scalar
            Value for Omega-Matter

        Omega_L : scalar
            Value for the cosmological constant (LAMBDA)
    
    Returns:
    --------
        pos
            3-D array containing the points comoving positions
    """

    # From degrees to radians
    RA = RA*np.pi/180
    Dec = np.pi/2-Dec*np.pi/180
    
    # Calculate R
    pool = mp.Pool(mp.cpu_count())
    R = pool.starmap_async(parallelize_AD, [(z, Omega_M, Omega_L) for z in redshift]).get()
    pool.close()
    
    pos = np.array([R*np.sin(Dec)*np.cos(RA), R*np.sin(Dec)*np.sin(RA),R*np.cos(Dec)])
    return pos.T
