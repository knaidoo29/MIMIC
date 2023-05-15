import os
import time

import numpy as np
import matplotlib.pylab as plt

from scipy.interpolate import interp1d

import fiesta
import magpie
import shift

from . import files
from . import read

from .. import theory
from .. import write
from .. import utils



mimic_beg = """
 ______________________________________________________________________________
|                   __  __   _____   __  __   _____    _____                   |
|                  |  \/  | |_   _| |  \/  | |_   _|  / ____|                  |
|                  | \  / |   | |   | \  / |   | |   | |                       |
|                  | |\/| |   | |   | |\/| |   | |   | |                       |
|                  | |  | |  _| |_  | |  | |  _| |_  | |____                   |
|                  |_|  |_| |_____| |_|  |_| |_____|  \_____|                  |
|______________________________________________________________________________|
|                                                                              |
|        Model Independent constrained cosMological Initial Conditions         |
|______________________________________________________________________________|

"""


mimic_end = """
 ______________________________________________________________________________
|                                                                              |
|                                 Finished                                     |
|______________________________________________________________________________|
"""


class MIMIC:


    def __init__(self, MPI):
        """Initialise the MIMIC main class."""
        # Global variables
        self.MPI = MPI
        self.FFT = None
        self.FFT_Ngrid = None
        self.ERROR = False

        # Time Variables
        self.time = {
            "Start": None,
            "End": None,
            "Prep_Start": None,
            "Prep_End": None,
            "WF_Start": None,
            "WF_End": None,
            "RZA_Start": None,
            "RZA_End": None
        }

        # Parameters
        self.params = None
        # Cosmology
        self.cosmo = {
            "H0": None,
            "Omega_m": None,
            "PowerSpecFile": None,
            "ScaleDepGrowth": None,
            "GrowthFile": None,
        }
        self.siminfo = {
            "Simname": None,
            "Boxsize": None,
            "Ngrid": None
        }
        self.constraints = {
            "Fname": None,
            "z_eff": None,
            "Rg": None,
            "CorrFile": None,
            "CovFile": None,
            "Sigma_NR": None,
            "Type": None
        }
        self.outputs = {
            "OutputFolder": None,
            "Prefix": None
        }
        self.ICs = {
            "Seed": None,
            "z_ic": None
        }
        self.what2run = {
            "WF": None,
            "RZA": None,
            "CR": None,
            "IC": None
        }
        # theoretical functions
        self.kmin = None
        self.kmax = None
        self.theory_kh = None
        self.theory_pk = None
        self.interp_pk = None
        self.growth_z = None
        self.growth_kh = None
        self.growth_Dzk = None
        self.growth_fzk = None
        self.interp_Dk = None
        self.interp_Dz = None
        # constraints file
        self.cons_x = None
        self.cons_y = None
        self.cons_z = None
        self.cons_ex = None
        self.cons_ey = None
        self.cons_ez = None
        self.cons_u = None
        self.cons_u_err = None
        # sim info
        self.halfsize = None
        self.sim_kmin = None
        self.sim_kmax = None
        self.x3D = None
        self.y3D = None
        self.z3D = None
        self.x_shape = None
        self.kx3D = None
        self.ky3D = None
        self.kz3D = None
        self.k_shape = None
        # correlators
        self.corr_r = None
        self.corr_xi = None
        self.corr_zeta = None
        self.corr_psiR = None
        self.corr_psiT = None
        self.interp_xi = None
        self.interp_zeta = None
        self.interp_psiR = None
        self.interp_psiT = None
        self.eta = None
        # store
        self.WF_dens = None
        # output
        self.fname_prefix = None


    def start(self):
        """Starts the run and timers."""
        self.time["Start"] = time.time()
        self.MPI.mpi_print_zero(mimic_beg)


    def _break4error(self):
        """Breaks the class run if an error is detected."""
        if self.ERROR is True:
            exit()


    def _get_fname_prefix(self):
        self.fname_prefix = self.outputs["OutputFolder"]
        if self.MPI.rank == 0:
            if write.check_folder_exist(self.fname_prefix) == False:
                write.create_folder(self.fname_prefix)
        if self.fname_prefix[-1] != "/":
            self.fname_prefix += "/"
        self.fname_prefix += self.outputs["Prefix"]
        return self.fname_prefix


    def _check_param_key(self, params, key):
        """Check param key exists in dictionary, and if key is not None."""
        if key in params:
            if params[key] is not None:
                return True
        else:
            return False


    def _check_exist(self, fname):
        if files.check_exist(fname) is False:
            self.MPI.mpi_print_zero(" ERROR: File '"+fname+"' does not exist, aborting.")
            self.ERROR = True
        self._break4error()


    def _bool2yesno(self, _x):
        if _x is True:
            return "Yes"
        else:
            return "No"


    def read_params(self, params):
        """Reads parameter file."""

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" Parameters")
        self.MPI.mpi_print_zero(" ==========")

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" MPI:")
        self.MPI.mpi_print_zero(" -", self.MPI.size, "Processors")

        # Read in Cosmological parameters
        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" Cosmology:")

        self.cosmo["H0"] = float(params["Cosmology"]["H0"])
        self.cosmo["Omega_m"] = float(params["Cosmology"]["Omega_m"])
        self.cosmo["PowerSpecFile"] = str(params["Cosmology"]["PowerSpecFile"])
        self.cosmo["ScaleDepGrowth"] = bool(params["Cosmology"]["ScaleDepGrowth"])
        self.cosmo["GrowthFile"] = str(params["Cosmology"]["GrowthFile"])

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" - H0 \t\t\t=", self.cosmo["H0"])
        self.MPI.mpi_print_zero(" - Omega_m \t\t=", self.cosmo["Omega_m"])
        self.MPI.mpi_print_zero(" - PowerSpecFile \t=", self.cosmo["PowerSpecFile"])
        self._check_exist(self.cosmo["PowerSpecFile"])
        self.MPI.mpi_print_zero(" - ScaleDepGrowth \t=", self.cosmo["ScaleDepGrowth"])
        self.MPI.mpi_print_zero(" - GrowthFile \t\t=", self.cosmo["GrowthFile"])
        self._check_exist(self.cosmo["GrowthFile"])

        # Read in Siminfo
        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" Siminfo:")

        self.siminfo["Simname"] = str(params["Siminfo"]["Simname"])
        self.siminfo["Boxsize"] = float(params["Siminfo"]["Boxsize"])
        self.siminfo["Ngrid"] = int(params["Siminfo"]["Ngrid"])

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" - Simname \t\t=", self.siminfo["Simname"])
        self.MPI.mpi_print_zero(" - Boxsize \t\t=", self.siminfo["Boxsize"])
        self.MPI.mpi_print_zero(" - Ngrid \t\t=", self.siminfo["Ngrid"])

        # Read in Constraints
        if self._check_param_key(params, "Constraints"):

            self.MPI.mpi_print_zero()
            self.MPI.mpi_print_zero(" Constraints:")

            self.constraints["Fname"] = str(params["Constraints"]["Fname"])
            self.constraints["z_eff"] = float(params["Constraints"]["z_eff"])
            self.constraints["Rg"] = float(params["Constraints"]["Rg"])
            self.constraints["CorrFile"] = params["Constraints"]["CorrFile"]
            self.constraints["CovFile"] = params["Constraints"]["CovFile"]
            self.constraints["Sigma_NR"] = float(params["Constraints"]["Sigma_NR"])
            self.constraints["Type"] = str(params["Constraints"]["Type"])

            self.MPI.mpi_print_zero()
            self._check_exist(self.constraints["Fname"])
            self.MPI.mpi_print_zero(" - Fname \t\t=", self.constraints["Fname"])
            self.MPI.mpi_print_zero(" - z_eff \t\t=", self.constraints["z_eff"])
            self.MPI.mpi_print_zero(" - Rg \t\t\t=", self.constraints["Rg"])
            if self.constraints["CorrFile"] is not None:
                self._check_exist(self.constraints["CorrFile"])
            self.MPI.mpi_print_zero(" - CorrFile \t\t=", self.constraints["CorrFile"])
            if self.constraints["CovFile"] is not None:
                self._check_exist(self.constraints["CovFile"])
            self.MPI.mpi_print_zero(" - CovFile \t\t=", self.constraints["CovFile"])
            self.MPI.mpi_print_zero(" - Sigma_NR \t\t=", self.constraints["Sigma_NR"])
            self.MPI.mpi_print_zero(" - Type \t\t=", self.constraints["Type"])

        # Outputs
        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" Outputs:")

        self.outputs["OutputFolder"] = str(params["Outputs"]["OutputFolder"])
        self.outputs["Prefix"] = str(params["Outputs"]["Prefix"])

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" - OutputFolder \t=", self.outputs["OutputFolder"])
        self.MPI.mpi_print_zero(" - Prefix \t\t=", self.outputs["Prefix"])

        # ICs
        if self._check_param_key(params, "ICs"):

            self.MPI.mpi_print_zero()
            self.MPI.mpi_print_zero(" ICs:")

            self.ICs["Seed"] = int(params["ICs"]["Seed"])
            self.ICs["z_ic"] = float(params["ICs"]["z_ic"])

            self.MPI.mpi_print_zero()
            self.MPI.mpi_print_zero(" - Seed \t\t=", self.ICs["Seed"])
            self.MPI.mpi_print_zero(" - z_ic \t\t=", self.ICs["z_ic"])

        # What to run
        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" Construct:")

        if self._check_param_key(params["Run"], "WF"):
            self.what2run["WF"] = bool(params["Run"]["WF"])
        else:
            self.what2run["WF"] = False

        if self._check_param_key(params["Run"], "RZA"):
            self.what2run["RZA"] = bool(params["Run"]["RZA"])
        else:
            self.what2run["RZA"] = False

        # RZA can only be performed in WF is computed
        if self.what2run["RZA"]:
            self.what2run["WF"] = True


        if self._check_param_key(params["Run"], "CR"):
            self.what2run["CR"] = bool(params["Run"]["CR"])
        else:
            self.what2run["CR"] = False

        if self._check_param_key(params["Run"], "IC"):
            self.what2run["IC"] = bool(params["Run"]["IC"])
        else:
            self.what2run["IC"] = False

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" - WF \t\t\t=", self._bool2yesno(self.what2run["WF"]))
        self.MPI.mpi_print_zero(" - RZA \t\t\t=", self._bool2yesno(self.what2run["RZA"]))
        self.MPI.mpi_print_zero(" - CR \t\t\t=", self._bool2yesno(self.what2run["CR"]))
        self.MPI.mpi_print_zero(" - IC \t\t\t=", self._bool2yesno(self.what2run["IC"]))


    def read_paramfile(self, yaml_fname):
        """Reads parameter file."""
        self.params, self.ERROR = read.read_paramfile(yaml_fname, self.MPI)
        if self.ERROR is False:
             self.read_params(self.params)
        self._break4error()


    def _get_interp_Dk(self, redshift):
        growth_Dk = np.zeros(len(self.growth_kh))
        for i in range(0, len(self.growth_kh)):
            interp_D = interp1d(self.growth_z, self.growth_Dzk[:, i], kind='cubic')
            growth_Dk[i] = interp_D(redshift)
        self.interp_Dk = interp1d(self.growth_kh, growth_Dk, kind='cubic', bounds_error=False)


    def _get_interp_Dz(self):
        self.interp_Dz = interp1d(self.growth_z, self.growth_Dz, kind='cubic')


    def _get_growth_D(self, redshift, kmag=None):
        if self.cosmo["ScaleDepGrowth"]:
            self._get_growth_Dk(redshift)
            return self.interp_Dk(kmag)
        else:
            self._get_interp_Dz()
            return self.interp_Dz(redshift)


    def _get_interp_fk(self, redshift):
        growth_fk = np.zeros(len(self.growth_kh))
        for i in range(0, len(self.growth_kh)):
            interp_f = interp1d(self.growth_z, self.growth_fzk[:, i], kind='cubic')
            growth_fk[i] = interp_f(redshift)
        self.interp_fk = interp1d(self.growth_kh, growth_fk, kind='cubic', bounds_error=False)


    def _get_interp_fz(self):
        self.interp_fz = interp1d(self.growth_z, self.growth_fz, kind='cubic')


    def _get_growth_f(self, redshift, kmag=None):
        if self.cosmo["ScaleDepGrowth"]:
            self._get_growth_fk(redshift)
            return self.interp_fk(kmag)
        else:
            self._get_interp_fz()
            return self.interp_fz(redshift)


    def prep_theory(self):
        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" Theory")
        self.MPI.mpi_print_zero(" ======")
        self.MPI.mpi_print_zero()

        self.MPI.mpi_print_zero(" - Load PowerSpecFile :", self.cosmo["PowerSpecFile"])
        data = np.load(self.cosmo["PowerSpecFile"])
        self.theory_kh, self.theory_pk = data['kh'], data['pk']

        self.MPI.mpi_print_zero(" - Create P(k) interpolator")
        self.kmin, self.kmax = self.theory_kh.min(), self.theory_kh.max()
        self.interp_pk = interp1d(self.theory_kh, self.theory_pk, kind='cubic', bounds_error=False, fill_value=0.)

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" - Load GrowthFile :", self.cosmo["GrowthFile"])
        self.MPI.mpi_print_zero(" - ScaleDepGrowth :", self.cosmo["ScaleDepGrowth"])

        if self.cosmo["ScaleDepGrowth"]:
            data = np.load(self.cosmo["GrowthFile"])
            growth_z, growth_Hz, growth_kh, growth_Dzk, growth_fzk = data['z'], data['Hz'], data['kh'], data['Dzk'], data['fzk']
        else:
            data = np.load(self.cosmo["GrowthFile"])
            growth_z, growth_Hz, growth_Dz, growth_fz = data['z'], data['Hz'], data['Dz'], data['fz']

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" - Create H(z) interpolator")
        self.interp_Hz = interp1d(growth_z, growth_Hz/(self.cosmo["H0"]*1e-2), kind='cubic')

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" - Store D(z,k) and f(z,k) for interpolation")

        if self.cosmo["ScaleDepGrowth"]:
            self.growth_z = growth_z
            self.growth_kh = growth_kh
            self.growth_Dzk = growth_Dzk
            self.growth_fzk = growth_fzk
        else:
            self.growth_z = growth_z
            self.growth_Dz = growth_Dz
            self.growth_fz = growth_fz


    def prep_constraints(self):
        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" Constraints")
        self.MPI.mpi_print_zero(" ===========")
        self.MPI.mpi_print_zero()
        # Basic properties of the sims
        self.halfsize = self.siminfo["Boxsize"]/2.
        # Load constraints
        self.MPI.mpi_print_zero(" - Load constraint file :", self.constraints["Fname"])
        data = np.load(self.constraints["Fname"])
        self.cons_x, self.cons_y, self.cons_z = data["x"], data["y"], data["z"]
        self.cons_ex, self.cons_ey, self.cons_ez = data["ex"], data["ey"], data["ez"]
        self.cons_u, self.cons_u_err = data["u"], data["u_err"]
        # Normalise direction
        self.MPI.mpi_print_zero(" - Normalize velocity unit vector")
        norm = np.sqrt(self.cons_ex**2. + self.cons_ey**2. + self.cons_ez**2.)
        self.cons_ex /= norm
        self.cons_ey /= norm
        self.cons_ez /= norm
        # Move position to the center of the box
        self.MPI.mpi_print_zero(" - Move constraints to the center of the simulation box")
        self.cons_x += self.halfsize
        self.cons_y += self.halfsize
        self.cons_z += self.halfsize
        # Keep only positions inside the box, r <= halfboxsize
        self.MPI.mpi_print_zero(" - Remove constrained points outside of the simulation box")
        r = np.sqrt((self.cons_x-self.halfsize)**2. + (self.cons_y-self.halfsize)**2. + (self.cons_z-self.halfsize)**2.)
        cond = np.where(r < self.halfsize)[0]
        self.MPI.mpi_print_zero(" - Retained %i constrained points from %i" % (len(cond), len(self.cons_x)))
        self.cons_x = self.cons_x[cond]
        self.cons_y = self.cons_y[cond]
        self.cons_z = self.cons_z[cond]
        self.cons_ex = self.cons_ex[cond]
        self.cons_ey = self.cons_ey[cond]
        self.cons_ez = self.cons_ez[cond]
        self.cons_u = self.cons_u[cond]
        self.cons_u_err = self.cons_u_err[cond]


    def prep_extra(self):
        self.SBX = fiesta.coords.MPI_SortByX(self.MPI)
        self.SBX.settings(self.siminfo["Boxsize"], self.siminfo["Ngrid"])
        self.SBX.limits4grid()


    def save_correlators(self):
        if self.MPI.rank == 0:
            fname_prefix = self._get_fname_prefix()
            fname = fname_prefix + "correlator_functions.npz"
            self.MPI.mpi_print_zero(" - Save correlation function as :", fname)
            np.savez(fname, r=self.corr_r, xi=self.corr_xi, zeta=self.corr_zeta, psiR=self.corr_psiR, psiT=self.corr_psiT)


    def calc_correlators(self, redshift):
        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" Correlators")
        self.MPI.mpi_print_zero(" ===========")
        self.MPI.mpi_print_zero()

        self.MPI.mpi_print_zero(" - Compute correlators in parallel")

        self.sim_kmin = shift.cart.get_kf(self.siminfo["Boxsize"])
        self.sim_kmax = shift.cart.get_kn(self.siminfo["Boxsize"], self.siminfo["Ngrid"])
        self.corr_r = np.logspace(-2, np.log10(np.sqrt(3.)*self.siminfo["Boxsize"]), 500)

        Dz2 = self._get_growth_D(redshift, kmag=self.theory_kh)**2.
        fz0 = self._get_growth_f(redshift, kmag=self.theory_kh)

        _corr_r = self.MPI.split_array(self.corr_r)
        _Rg = self.constraints["Rg"]

        _xi = theory.pk2xi(_corr_r, self.theory_kh, Dz2*self.theory_pk, kmin=self.sim_kmin,
            kmax=self.sim_kmax, kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6), Rg=_Rg)
        _zeta = theory.pk2zeta(_corr_r, self.theory_kh, Dz2*self.theory_pk, fk=fz0, kmin=self.sim_kmin,
            kmax=self.sim_kmax, kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6), Rg=_Rg, cons_type=self.constraints["Type"])
        _psiR = theory.pk2psiR(_corr_r, self.theory_kh, Dz2*self.theory_pk, fk=fz0, kmin=self.sim_kmin,
            kmax=self.sim_kmax, kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6), Rg=_Rg, cons_type=self.constraints["Type"])
        _psiT = theory.pk2psiT(_corr_r, self.theory_kh, Dz2*self.theory_pk, fk=fz0, kmin=self.sim_kmin,
            kmax=self.sim_kmax, kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6), Rg=_Rg, cons_type=self.constraints["Type"])

        self.MPI.wait()

        self.MPI.mpi_print_zero(" - Collect correlation functions")

        _corr_r = self.MPI.collect(_corr_r)
        _xi = self.MPI.collect(_xi)
        _zeta = self.MPI.collect(_zeta)
        _psiR = self.MPI.collect(_psiR)
        _psiT = self.MPI.collect(_psiT)

        self.MPI.mpi_print_zero(" - Broadcast correlation functions")

        _corr_r = self.MPI.broadcast(_corr_r)
        _xi = self.MPI.broadcast(_xi)
        _zeta = self.MPI.broadcast(_zeta)
        _psiR = self.MPI.broadcast(_psiR)
        _psiT = self.MPI.broadcast(_psiT)

        self.corr_r = np.concatenate([np.array([0.]), _corr_r])
        self.corr_xi = np.concatenate([np.array([_xi[0]]), _xi])
        self.corr_zeta = np.concatenate([np.array([0.]), _zeta])
        self.corr_psiR = np.concatenate([np.array([_psiR[0]]), _psiR])
        self.corr_psiT = np.concatenate([np.array([_psiT[0]]), _psiT])

        self.MPI.mpi_print_zero(" - Construct interpolators")

        self.interp_xi = interp1d(self.corr_r, self.corr_xi, kind='cubic', bounds_error=False, fill_value=0.)
        self.interp_zeta = interp1d(self.corr_r, self.corr_zeta, kind='cubic', bounds_error=False, fill_value=0.)
        self.interp_psiR = interp1d(self.corr_r, self.corr_psiR, kind='cubic', bounds_error=False, fill_value=0.)
        self.interp_psiT = interp1d(self.corr_r, self.corr_psiT, kind='cubic', bounds_error=False, fill_value=0.)

        self.save_correlators()


    def compute_eta(self):
        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" Compute eta-vector")
        self.MPI.mpi_print_zero(" ==================")
        self.MPI.mpi_print_zero()

        x1, x2 = self.MPI.create_split_ndgrid([self.cons_x, self.cons_x], [False, True])
        y1, y2 = self.MPI.create_split_ndgrid([self.cons_y, self.cons_y], [False, True])
        z1, z2 = self.MPI.create_split_ndgrid([self.cons_z, self.cons_z], [False, True])

        ex1, ex2 = self.MPI.create_split_ndgrid([self.cons_ex, self.cons_ex], [False, True])
        ey1, ey2 = self.MPI.create_split_ndgrid([self.cons_ey, self.cons_ey], [False, True])
        ez1, ez2 = self.MPI.create_split_ndgrid([self.cons_ez, self.cons_ez], [False, True])

        psiT0 = self.corr_psiT[0]

        self.MPI.mpi_print_zero(" - Computer vel-vel covariance matrix in parallel")

        cov_uu = theory.get_cov_uu(x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2,
            ez1, ez2, self.interp_psiR, self.interp_psiT, self.constraints["z_eff"],
            self.interp_Hz, psiT0, cons_type=self.constraints["Type"])

        self.MPI.mpi_print_zero(" - Collect vel-vel covariance matrix [at MPI.rank = 0]")

        cov_uu = self.MPI.collect(cov_uu)

        if self.MPI.rank == 0:
            cov_uu += np.diag(self.cons_u_err**2.)
            # add sigma_NR more error?

            self.MPI.mpi_print_zero(" - Inverting matrix [at MPI.rank = 0]")
            inv_uu = np.linalg.inv(cov_uu)

            self.MPI.mpi_print_zero(" - Compute eta vector [at MPI.rank = 0]")
            self.eta = inv_uu.dot(self.cons_u)

        self.MPI.wait()

        self.MPI.mpi_print_zero(" - Broadcast eta vector")
        self.eta = self.MPI.broadcast(self.eta)


    def get_grid3D(self):
        self.MPI.mpi_print_zero(" - Construct cartesian grid")
        self.x3D, self.y3D, self.z3D = shift.cart.mpi_grid3D(self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)


    def flatten_grid3D(self):
        self.x_shape = np.shape(self.x3D)
        self.x3D = self.x3D.flatten()
        self.y3D = self.y3D.flatten()
        self.z3D = self.z3D.flatten()


    def unflatten_grid3D(self):
        self.x3D = self.x3D.reshape(self.x_shape)
        self.y3D = self.y3D.reshape(self.x_shape)
        self.z3D = self.z3D.reshape(self.x_shape)


    def get_grid_mag(self):
        return np.sqrt(self.x3D**2. + self.y3D**2. + self.z3D**2.)


    def get_kgrid3D(self):
        self.MPI.mpi_print_zero(" - Construct Fourier grid")
        self.kx3D, self.ky3D, self.kz3D = shift.cart.mpi_kgrid3D(self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)
        self.k_shape = np.shape(self.kx3D)


    def flatten_kgrid3D(self):
        self.kx3D = self.kx3D.flatten()
        self.ky3D = self.ky3D.flatten()
        self.kz3D = self.kz3D.flatten()


    def unflatten_kgrid3D(self):
        self.kx3D = self.kx3D.reshape(self.k_shape)
        self.ky3D = self.ky3D.reshape(self.k_shape)
        self.kz3D = self.kz3D.reshape(self.k_shape)


    def get_kgrid_mag(self):
        return np.sqrt(self.kx3D**2. + self.ky3D**2. + self.kz3D**2.)


    def _get_WF_single(self, x, y, z, adot, i, total):
        rx = self.cons_x - x
        ry = self.cons_y - y
        rz = self.cons_z - z
        r = np.sqrt(rx**2. + ry**2. + rz**2.)
        norm_rx = np.copy(rx)/r
        norm_ry = np.copy(ry)/r
        norm_rz = np.copy(rz)/r
        cov_du = self.interp_zeta(r)
        du = - adot*cov_du*norm_rx*self.cons_ex - adot*cov_du*norm_ry*self.cons_ey - adot*cov_du*norm_rz*self.cons_ez
        if self.MPI.rank == 0:
            utils.progress_bar(i, total, explanation=" ---- ", indexing=True)
        return du.dot(self.eta)


    def save_WF(self, WF_dens):
        fname_prefix = self._get_fname_prefix()
        fname = fname_prefix + "WF_"+str(self.MPI.rank)+".npz"
        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" - Save WF :", fname_prefix+"WF_[0-%i].npz" % (self.MPI.size-1))
        np.savez(fname, x3D=self.x3D, y3D=self.y3D, z3D=self.z3D, WF_dens=WF_dens)


    def get_WF(self):
        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" Compute Wiener Filter")
        self.MPI.mpi_print_zero(" =====================")
        self.MPI.mpi_print_zero()

        self.get_grid3D()
        self.flatten_grid3D()

        z0 = self.constraints["z_eff"]
        Hz = self.interp_Hz(z0)

        if self.constraints["Type"] == "Vel":
            adot = theory.z2a(z0)*Hz
        elif self.constraints["Type"] == "Psi":
            adot = 1.

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" - Computing Wiener Filter density")
        WF_dens = np.array([self._get_WF_single(self.x3D[i], self.y3D[i], self.z3D[i], adot, i, len(self.x3D)) for i in range(0, len(self.x3D))])

        self.unflatten_grid3D()
        WF_dens = WF_dens.reshape(self.x_shape)

        self.save_WF(WF_dens)

        if self.what2run["RZA"]:
            self.WF_dens = WF_dens


    def start_FFT(self, Ngrid):
        if self.FFT is None or self.FFT_Ngrid != Ngrid:
            self.FFT_Ngrid = Ngrid
            Ngrids = np.array([Ngrid, Ngrid, Ngrid], dtype=int)
            self.FFT = self.MPI.mpi_fft_start(Ngrids)


    def complex_zeros(self, shape):
        return np.zeros(shape) + 1j*np.zeros(shape)


    def dens2psi(self, dens):
        self.start_FFT(self.siminfo["Ngrid"])
        self.get_kgrid3D()
        kmag = self.get_kgrid_mag()
        densk = shift.cart.mpi_fft3D(dens, self.x_shape, self.siminfo["Boxsize"],
            self.siminfo["Ngrid"], self.FFT)
        psi_kx = self.complex_zeros(self.k_shape)
        psi_ky = self.complex_zeros(self.k_shape)
        psi_kz = self.complex_zeros(self.k_shape)
        cond = np.where(kmag != 0.)
        psi_kx[cond] = densk[cond] * 1j * self.kx3D[cond]/(kmag[cond]**2.)
        psi_ky[cond] = densk[cond] * 1j * self.ky3D[cond]/(kmag[cond]**2.)
        psi_kz[cond] = densk[cond] * 1j * self.kz3D[cond]/(kmag[cond]**2.)
        psi_x = shift.cart.mpi_ifft3D(psi_kx, self.x_shape, self.siminfo["Boxsize"],
            self.siminfo["Ngrid"], self.FFT)
        psi_y = shift.cart.mpi_ifft3D(psi_ky, self.x_shape, self.siminfo["Boxsize"],
            self.siminfo["Ngrid"], self.FFT)
        psi_z = shift.cart.mpi_ifft3D(psi_kz, self.x_shape, self.siminfo["Boxsize"],
            self.siminfo["Ngrid"], self.FFT)
        return psi_x, psi_y, psi_z


    def psi2vel(self, redshift, psi_x, psi_y, psi_z):
        #z0 = self.constraints["z_eff"]
        z0 = redshift
        Hz = self.interp_Hz(z0)

        if self.constraints["Type"] == "Vel":
            adot = theory.z2a(z0)*Hz
        elif self.constraints["Type"] == "Psi":
            adot = 1.

        if self.cosmo["ScaleDepGrowth"]:
            self.start_FFT(self.siminfo["Ngrid"])
            self.get_kgrid3D()
            kmag = self.get_kgrid_mag()

            vel_kx = shift.cart.mpi_fft3D(psi_x, self.x_shape, self.siminfo["Boxsize"],
                self.siminfo["Ngrid"], self.FFT)
            vel_ky = shift.cart.mpi_fft3D(psi_y, self.x_shape, self.siminfo["Boxsize"],
                self.siminfo["Ngrid"], self.FFT)
            vel_kz = shift.cart.mpi_fft3D(psi_z, self.x_shape, self.siminfo["Boxsize"],
                self.siminfo["Ngrid"], self.FFT)

            cond = np.where(kmag != 0.)
            fk = self._get_growth_f(z0, kmag=kmag[cond])
            vel_kx[cond] = adot*fk*psi_kx[cond]
            vel_ky[cond] = adot*fk*psi_ky[cond]
            vel_kz[cond] = adot*fk*psi_kz[cond]

            vel_x = shift.cart.mpi_ifft3D(vel_kx, self.x_shape, self.siminfo["Boxsize"],
                self.siminfo["Ngrid"], self.FFT)
            vel_y = shift.cart.mpi_ifft3D(vel_ky, self.x_shape, self.siminfo["Boxsize"],
                self.siminfo["Ngrid"], self.FFT)
            vel_z = shift.cart.mpi_ifft3D(vel_kz, self.x_shape, self.siminfo["Boxsize"],
                self.siminfo["Ngrid"], self.FFT)

        else:
            fz = self._get_growth_f(z0)
            vel_x = adot*fz*psi_x
            vel_y = adot*fz*psi_y
            vel_z = adot*fz*psi_z

        return vel_x, vel_y, vel_z


    def add_buffer_in_x(self, f):
        f_send_down = self.MPI.send_down(f[0])
        f_send_up = self.MPI.send_up(f[-1])
        fnew = np.concatenate([np.array([f_send_up]), f, np.array([f_send_down])])
        return fnew

    def get_buffer_range(self):
        dx = self.siminfo["Boxsize"]/self.siminfo["Ngrid"]
        xmin = np.min(self.x3D) - dx/2. - dx
        xmax = np.max(self.x3D) + dx/2. + dx
        return xmin, xmax

    def unbuffer_in_x(self, f):
        fnew = f[1:-1]
        return fnew

    def get_RZA(self):
        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" Apply Reverse Zeldovich Approximation")
        self.MPI.mpi_print_zero(" =====================================")
        self.MPI.mpi_print_zero()

        self.MPI.mpi_print_zero(" - Converting WF density to displacement Psi")
        self.MPI.mpi_print_zero()

        psi_x, psi_y, psi_z = self.dens2psi(self.WF_dens)
        psi_x = self.add_buffer_in_x(psi_x)
        psi_y = self.add_buffer_in_x(psi_y)
        psi_z = self.add_buffer_in_x(psi_z)
        xmin, xmax = self.get_buffer_range()

        if self.MPI.rank == 0:
            data = np.column_stack([self.cons_x, self.cons_y, self.cons_z, self.cons_ex,
                self.cons_ey, self.cons_ez, self.cons_u, self.cons_u_err])
        else:
            data = None
        self.SBX.input(data)
        data = self.SBX.distribute()
        cons_x, cons_y, cons_z, cons_ex, cons_ey, cons_ez, cons_u, cons_u_err = \
            data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5], data[:,6], data[:,7]

        self.MPI.mpi_print_zero(" - Interpolating displacement Psi at constraint points")
        self.MPI.mpi_print_zero()

        if len(cons_x) != 0:
            cons_psi_x = fiesta.interp.trilinear(psi_x, [xmax-xmin, self.siminfo["Boxsize"], self.siminfo["Boxsize"]],
                cons_x, cons_y, cons_z, origin=[xmin, 0., 0.], periodic=[False, True, True])
            cons_psi_y = fiesta.interp.trilinear(psi_y, [xmax-xmin, self.siminfo["Boxsize"], self.siminfo["Boxsize"]],
                cons_x, cons_y, cons_z, origin=[xmin, 0., 0.], periodic=[False, True, True])
            cons_psi_z = fiesta.interp.trilinear(psi_z, [xmax-xmin, self.siminfo["Boxsize"], self.siminfo["Boxsize"]],
                cons_x, cons_y, cons_z, origin=[xmin, 0., 0.], periodic=[False, True, True])
        else:
            cons_psi_x, cons_psi_y, cons_psi_z = None, None, None
        self.MPI.wait()

        self.MPI.mpi_print_zero(" - Applying RZA")
        self.MPI.mpi_print_zero()

        cons_rza_x = cons_x - cons_psi_x
        cons_rza_y = cons_y - cons_psi_y
        cons_rza_z = cons_z - cons_psi_z
        cons_rza_x -= self.halfsize
        cons_rza_y -= self.halfsize
        cons_rza_z -= self.halfsize
        # This assumes Method II of https://theses.hal.science/tel-01127294/document see page 121
        cons_rza_ex = np.copy(cons_ex)
        cons_rza_ey = np.copy(cons_ey)
        cons_rza_ez = np.copy(cons_ez)
        cons_rza_u = np.copy(cons_u)
        cons_rza_u_err = np.copy(cons_u_err)

        cons_rza_x = self.MPI.collect(cons_rza_x)
        cons_rza_y = self.MPI.collect(cons_rza_y)
        cons_rza_z = self.MPI.collect(cons_rza_z)

        cons_rza_ex = self.MPI.collect(cons_rza_ex)
        cons_rza_ey = self.MPI.collect(cons_rza_ey)
        cons_rza_ez = self.MPI.collect(cons_rza_ez)

        cons_rza_u = self.MPI.collect(cons_rza_u)
        cons_rza_u_err = self.MPI.collect(cons_rza_u_err)

        fname = self._get_fname_prefix() + 'rza.npz'
        self.MPI.mpi_print_zero(" - Saving RZA constraints to: %s" % fname)
        self.MPI.mpi_print_zero()
        if self.MPI.rank == 0:
            np.savez(fname, x=cons_rza_x, y=cons_rza_y, z=cons_rza_z,
                ex=cons_ex, ey=cons_ey, ez=cons_ez, u=cons_rza_u, u_err=cons_rza_u_err)

    ## HERE ##


    def run(self, yaml_fname):
        """Run MIMIC."""
        self.start()
        self.read_paramfile(yaml_fname)
        # Theory
        self.time["Prep_Start"] = time.time()
        self.prep_theory()
        self.prep_constraints()
        self.prep_extra()
        self.calc_correlators(self.constraints["z_eff"])
        self.compute_eta()
        self.time["Prep_End"] = time.time()

        if self.what2run["WF"]:
            self.time["WF_Start"] = time.time()
            self.get_WF()
            self.time["WF_End"] = time.time()

        if self.what2run["WF"]:
            self.time["RZA_Start"] = time.time()
            self.get_RZA()
            self.time["RZA_End"] = time.time()
        self.end()


    def _print_time(self, prefix, time):
        """Compute print time.

        Parameters
        ----------
        prefix: str
            Prefix to time ouptut.
        time : float
            Time.
        """
        if time < 1.:
            self.MPI.mpi_print_zero(prefix, "%0.6f seconds" % time)
        elif time < 60:
            self.MPI.mpi_print_zero(prefix, "%0.2f seconds" % time)
        elif time < 60*60:
            time /= 60
            self.MPI.mpi_print_zero(prefix, "%0.2f minutes" % time)
        else:
            time /= 60*60
            self.MPI.mpi_print_zero(prefix, "%0.2f hours" % time)


    def end(self):
        """Ends the run."""
        self.MPI.wait()
        self.time["End"] = time.time()

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" Running Time")
        self.MPI.mpi_print_zero(" ============")
        self.MPI.mpi_print_zero()

        self._print_time(" -> Preparation\t\t= ", self.time["Prep_End"] - self.time["Prep_Start"])
        if self.what2run["WF"]:
            self._print_time(" -> Wiener Filter\t= ", self.time["WF_End"] - self.time["WF_Start"])
        if self.what2run["RZA"]:
            self._print_time(" -> Rev. Zel. Approx.\t= ", self.time["RZA_End"] - self.time["RZA_Start"])

        self.MPI.mpi_print_zero()
        self._print_time(" -> Total\t\t= ", self.time["End"] - self.time["Start"])

        self.MPI.mpi_print_zero(mimic_end)
