import numpy as np


def z2a(z):
    """Redshift to scale factor."""
    return 1./(1.+z)


def a2z(a):
    """Scale factor to redshift."""
    return (1./a) - 1.



class BackgroundCosmology:

    """Class for computing the Background densities and evolution for the
    various cosmological species."""


    def __init__(self):
        """Initialises the class and basic properties."""
        # Functional variables.
        self.verbose = True
        # Hubble Constant
        self.H0 = None
        self.h = None
        # Neutrinos
        self.mnu = None
        self.Neff = None
        self.Nnu = None
        self.Nnu_NR = None
        self.a_NRs = None
        self.use_nu = False
        self.nu_mass_split = None
        # Other Species
        self.Omega_b0 = None
        self.use_b = False
        self.Omega_cdm0 = None
        self.use_cdm = False
        self.Omega_m0 = None
        self.Omega_r0 = None
        self.Omega_L0 = None


    # Hubble Constant
    def set_H0(self, H0):
        """Sets the value of the Hubble constant."""
        self.H0 = H0
        self.h = H0/100.


    # Photons
    def get_Omega_g(self, a):
        """Returns the photon density at scale factor a.

        Parameters
        ----------
        a : float or array
            Scale factor.
        """
        return 0.247282/((self.H0**2.)*(a**4.))


    def get_rho_g(self, a):
        """Returns the photon density in units of 8piG/3 at a given scale factor a."""
        return self.get_Omega_g(a)*(self.H0**2.)


    # Neutrinos
    def when_nu_nonrel(self, mnu):
        """Determines when a neutrino species becomes non-relativistic.

        Parameters
        ----------
        mnu : float
            Mass of a single neutrino species.

        Returns
        -------
        a_NR : float
            Scale factor at which the neutrino species becomes non-relativistic.
        """
        if mnu > 0.:
            a_NR = (self.Neff/self.Nnu)*((7./8.)*((4./11.)**(4./3.)))*0.247282*9.314e-3/mnu
        else:
            a_NR = np.inf # Never will become non-relativistic
        return a_NR


    def set_nu(self, mnu, Neff=3.046, Nnu=3, nu_mass_split=[1., 0., 0.]):
        """Set neutrino mass.

        Parameters
        ----------
        mnu : float
            The sum of the masses of neutrino species.
        Neff : float
            Effective number of neutrino species.
        Nnu : int
            Number of neutrino species.
        mass_split : list
            How to split the mass between species.
        """
        # Count number of massive neutrinos
        self.use_nu = True
        self.mnu = mnu
        self.Neff = Neff
        self.Nnu = Nnu
        self.nu_mass_split = nu_mass_split
        self.Nnu_NR = 0
        self.a_NRs = []
        for i in range(0, len(self.nu_mass_split)):
            if self.nu_mass_split[i] != 0.:
                self.Nnu_NR += 1
            a_NR = self.when_nu_nonrel(self.nu_mass_split[i]*mnu)
            self.a_NRs.append(a_NR)


    def get_Omega_nu_single_R(self, a, a_NR):
        """Returns the relativistic neutrino density for a single species.

        Parameters
        ----------
        a : float or array
            Scale factor.
        a_NR : float
            Scale factor transition to non-relativistic species.

        Returns
        -------
        Omega_nu_R : float or array
            Relativistic neutrino density for single species.
        """
        Omega_nu_const = (self.Neff/self.Nnu)*((7./8.)*((4./11.)**(4./3.)))
        if np.isscalar(a) == True:
            if a < a_NR:
                return Omega_nu_const*self.get_Omega_g(a)
            else:
                return 0.
        else:
            Omega_nu_R = np.zeros(len(a))
            cond = np.where(a < a_NR)[0]
            Omega_nu_R[cond] = Omega_nu_const*self.get_Omega_g(a[cond])
            return Omega_nu_R


    def get_Omega_nu_single_NR(self, a, mnu, a_NR):
        """Returns the non-relativistic neutrino density for a single species.

        Parameters
        ----------
        a : float or array
            Scale factor.
        mnu : float
            Mass of neutrino species.
        a_NR : float
            Scale factor transition to non-relativistic species.

        Returns
        -------
        Omega_nu_NR : float or array
            Non-relativistic neutrino density for single species."""
        Omega_nu_const = mnu/((9.314e-3)*(self.H0**2.))
        if np.isscalar(a) == True:
            if a < a_NR:
                return 0.
            else:
                return Omega_nu_const/(a**3.)
        else:
            Omega_nu_NR = np.zeros(len(a))
            cond = np.where(a >= a_NR)[0]
            Omega_nu_NR[cond] = Omega_nu_const/(a[cond]**3.)
        return Omega_nu_NR


    def get_Omega_nu(self, a, output='both'):
        """Returns the non-relativistic neutrino density for a single species.

        Parameters
        ----------
        a : float or array
            Scale factor.
        mnu : float
            Mass of neutrino species.
        a_NR : float
            Scale factor transition to non-relativistic species.

        Returns
        -------
        Omega_nu_R : float or array
            Relativistic neutrino density.
        Omega_nu_NR : float or array
            Non-relativistic neutrino density.
        """
        if self.use_nu is True:
            for i in range(0, len(self.a_NRs)):
                if output == 'both' or output == 'R':
                    _Omega_nu_R = self.get_Omega_nu_single_R(a, self.a_NRs[i])
                if output == 'both' or output == 'NR':
                    _Omega_nu_NR = self.get_Omega_nu_single_NR(a, self.mnu*self.nu_mass_split[i], self.a_NRs[i])
                if i == 0:
                    if output == 'both' or output == 'R':
                        Omega_nu_R = _Omega_nu_R
                    if output == 'both' or output == 'NR':
                        Omega_nu_NR = _Omega_nu_NR
                else:
                    if output == 'both' or output == 'R':
                        Omega_nu_R += _Omega_nu_R
                    if output == 'both' or output == 'NR':
                        Omega_nu_NR += _Omega_nu_NR
        else:
            Omega_nu_R, Omega_nu_NR = 0., 0.
        if output == 'both':
            return Omega_nu_R, Omega_nu_NR
        if output == 'R':
            return Omega_nu_R
        if output == 'NR':
            return Omega_nu_NR


    def get_rho_nu(self, a, output):
        """Returns the neutrino density in units of 8piG/3 at a given scale factor a."""
        if output == 'both':
            Omega_nu_R, Omega_nu_NR = self.get_Omega_nu(a, output=output)
            return Omega_nu_R*(self.H0**2.), Omega_nu_NR*(self.H0**2.)
        if output == 'R' or output == 'NR':
            return self.get_Omega_nu(a, output=output)*(self.H0**2.)


    # Total Radiation
    def get_Omega_R(self, a):
        """Returns the total Radiation density for a given scale factor."""
        return self.get_Omega_g(a) + self.get_Omega_nu(a, output='R')


    def get_rho_R(self, a):
        """Returns the total Radiation density in units of 8piG/3 for a given scale factor."""
        return self.get_Omega_R(a)*(self.H0**2.)


    # Baryons
    def set_baryons(self, Omega_b0):
        """Sets the amount of baryonic matter."""
        self.use_b = True
        self.Omega_b0 = Omega_b0


    def get_Omega_b(self, a):
        """Returns the baryon density for a given scale factor."""
        if self.use_b is True:
            return self.Omega_b0/(a**3.)
        else:
            return 0.


    def get_rho_b(self, a):
        """Returns the total baryon density in units of 8piG/3 for a given scale factor."""
        return self.get_Omega_b(a)*(self.H0**2.)


    # Cold Dark Matter
    def set_cdm(self, Omega_cdm0):
        """Sets the amount of cold dark matter."""
        self.use_cdm = True
        self.Omega_cdm0 = Omega_cdm0


    def get_Omega_cdm(self, a):
        """Returns the cold dark matter density for a given scale factor."""
        if self.use_cdm is True:
            return self.Omega_cdm0/(a**3.)
        else:
            return 0.


    def get_rho_cdm(self, a):
        """Returns the total cold dark matter density in units of 8piG/3 for a given scale factor."""
        return self.get_Omega_cdm(a)*(self.H0**2.)


    # Total Dark Matter - cold
    def get_Omega_dm(self, a):
        """Returns the total dark matter density for a given scale factor."""
        return self.get_Omega_cdm(a)


    def get_rho_dm(self, a):
        """Returns the total dark matter in units of 8piG/3 for a given scale factor."""
        return self.get_Omega_dm(a)*(self.H0**2.)


    # Total Matter
    def get_Omega_M(self, a):
        """Returns the total amount of matter density for a given scale factor."""
        return self.get_Omega_cdm(a) + self.get_Omega_b(a) + self.get_Omega_nu(a, output='NR')


    def get_rho_M(self, a):
        """Returns the total matter in units of 8piG/3 for a given scale factor."""
        return self.get_Omega_M(a)*(self.H0**2.)


    # Lambda
    def set_Lambda(self):
        """Sets the energy density for Lambda assuming zero curvature."""
        Omega_nu_R0, Omega_nu_NR0 = self.get_Omega_nu(1.)
        self.Omega_r0 = self.get_Omega_g(1.) + Omega_nu_R0
        self.Omega_m0 = self.get_Omega_cdm(1.) + self.get_Omega_b(1.) + Omega_nu_NR0
        self.Omega_L0 = 1. - self.Omega_r0 - self.Omega_m0


    def get_Omega_L(self, a):
        """Returns dark energy (assuming a cosmological constant) density for a given scale factor."""
        if np.isscalar(a) is True:
            return self.Omega_L0
        else:
            return self.Omega_L0*np.ones(len(a))


    def get_rho_L(self, a):
        """Returns the total dark energy density in units of 8piG/3 for a given scale factor."""
        return self.get_Omega_L(a)*(self.H0**2.)


    # Total
    def get_Omega_T(self, a):
        """Returns the total density of radiation, matter and the cosmological constant."""
        return self.get_Omega_R(a) + self.get_Omega_M(a) + self.get_Omega_L(a)


    def get_rho_T(self, a):
        """Returns the total density of species in units of 8piG/3 for a given scale factor."""
        return self.get_Omega_T(a)*(self.H0**2.)


    # Friedmann equations
    def get_E(self, a):
        """Returns the normalised Hubble expansion function for a given scale factor."""
        return np.sqrt(self.get_Omega_T(a))


    # Hubble Parameter
    def get_H(self, a):
        """Returns the Hubble expansion rate (the first Friedmann equation) at a
        given scale factor."""
        return self.H0*self.get_E(a)


    def clean(self):
        """Reinitialises the class."""
        self.__init__()



def get_Hz_LCDM(redshift, H0, Omega_cdm0, Omega_b0=0., mnu=0., Neff=3.046,
    Nnu=3, nu_mass_split=[1., 0., 0.]):
    """Precise Hubble expansion rate valid at late and early times. Can be used
    to specify massive neutrinos. The function assume curvature is zero and sets
    Lambda so that this is true.

    Parameters
    ----------
    redshift : float or array
        Redshift to which you want the LCDM Hubble expansion rate.
    Omega_cdm0: float
        The amount of cold dark matter, if Omega_b0=0 this is the total matter
        content in the Universe.
    Omega_b0 : float, optional
        The baryon energy density fraction.
    mnu : float, optional
        Total mass of neutrinos.
    Neff : float, optional
        The effective number of neutrino species.
    Nnu : int, optional
        The number of neutrino species.
    nu_mass_split : list, optional
        The mass splitting for each neutrino species.

    Returns
    -------
    Hz : array or float
        Hubble expansion at the specified redshifts.
    """
    BC = BackgroundCosmology()
    BC.set_H0(H0)
    if mnu > 0.:
        BC.set_nu(mnu, Neff=Neff, Nnu=Nnu, nu_mass_split=nu_mass_split)
    if Omega_b0 > 0:
        BC.set_baryons(Omega_b0)
    BC.set_cdm(Omega_cdm0)
    BC.set_Lambda()
    a = z2a(redshift)
    Hz = BC.get_H(a)
    return Hz
