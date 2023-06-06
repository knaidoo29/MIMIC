import os
import sys
import time

import numpy as np

from scipy.interpolate import interp1d

from ..ext import fiesta, shift

from .. import field, io, src, theory


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


    def __init__(self, MPI=None):
        """Initialise the MIMIC main class."""

        # Global variables
        self.MPI = MPI
        self.FFT = None
        self.FFT_Ngrid = None
        self.ERROR = False

        # Added to *eventually* include no MPI functionality. This may not be
        # added, as it's unclear whether this will be necessary in the long run.
        if self.MPI is not None:
            self.rank = self.MPI.rank
            self.nompi = False
        else:
            self.rank = 0
            self.nompi = True

        # Time Variables
        self.time = {
            "Start": None,
            "End": None,
            "Prep_Start": None,
            "Prep_End": None,
            "WF_Start": None,
            "WF_End": None,
            "RZA_Start": None,
            "RZA_End": None,
            "CR_Prep_Start": None,
            "CR_Prep_End": None,
            "CR_Start": None,
            "CR_End": None
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
            "Boxsize": None,
            "Ngrid": None
        }
        self.constraints = {
            "Fname": None,
            "z_eff": None,
            "Rg": None,
            "CorrFile": None,
            "CovFile": None,
            "Sigma_NL": None,
            "Type": "Vel",
            "RZA_Method": 2
        }
        self.outputs = {
            "OutputFolder": None,
            "Prefix": None
        }
        self.ICs = {
            "Seed": None,
            "WNFile": None,
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
        self.cons_c = None
        self.cons_c_err = None
        self.cons_c_type = None
        self.cons_c_RR = None
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
        self.corr_redshift = None
        self.corr_r = None
        self.corr_xi = None
        self.corr_zeta_p = None
        self.corr_zeta_u = None
        self.corr_psiR_pp = None
        self.corr_psiT_pp = None
        self.corr_psiR_pu = None
        self.corr_psiT_pu = None
        self.corr_psiR_uu = None
        self.corr_psiT_uu = None
        self.interp_xi = None
        self.interp_zeta_p = None
        self.interp_zeta_u = None
        self.interp_psiR_pp = None
        self.interp_psiT_pp = None
        self.interp_psiR_pu = None
        self.interp_psiT_pu = None
        self.interp_psiR_uu = None
        self.interp_psiT_uu = None
        self.cov = None
        self.inv = None
        self.cov_CR = None
        self.inv_CR = None
        self.eta = None
        self.eta_CR = None
        # store
        self.dens_WF = None
        self.phi_x_WF = None
        self.phi_y_WF = None
        self.phi_z_WF = None
        self.vel_x_WF = None
        self.vel_y_WF = None
        self.vel_z_WF = None
        self.dens = None
        self.phi_x = None
        self.phi_y = None
        self.phi_z = None
        self.vel_x = None
        self.vel_y = None
        self.vel_z = None
        self._lenpro = 20
        # output
        self.fname_prefix = None


    def _print_zero(self, *value):
        """Print at rank=0."""
        if self.nompi is None:
            print(*value, flush=True)
        else:
            self.MPI.mpi_print_zero(*value)


    def start(self):
        """Starts the run and timers."""
        self.time["Start"] = time.time()
        self._print_zero(mimic_beg)


    def _break4error(self):
        io._break4error(self.ERROR)


    def _get_fname_prefix(self):
        self.fname_prefix = self.outputs["OutputFolder"]
        if self.rank == 0:
            if io.isfolder(self.fname_prefix) == False:
                io.create_folder(self.fname_prefix)
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
        if io.isfile(fname) is False:
            self.ERROR = True
        io._error_message(self.ERROR, "File %s does not exist" % fname, MPI=self.MPI)
        self._break4error()


    def _read_params(self, params):
        """Reads parameter file."""

        self._print_zero()
        self._print_zero(" Parameters")
        self._print_zero(" ==========")

        if self.nompi is False:
            self._print_zero()
            self._print_zero(" MPI:")
            self._print_zero(" -", self.MPI.size, "Processors")

        # Read in Cosmological parameters
        self._print_zero()
        self._print_zero(" Cosmology:")

        self.cosmo["H0"] = float(params["Cosmology"]["H0"])
        self.cosmo["Omega_m"] = float(params["Cosmology"]["Omega_m"])
        self.cosmo["PowerSpecFile"] = str(params["Cosmology"]["PowerSpecFile"])
        self.cosmo["ScaleDepGrowth"] = bool(params["Cosmology"]["ScaleDepGrowth"])
        self.cosmo["GrowthFile"] = str(params["Cosmology"]["GrowthFile"])

        self._print_zero()
        self._print_zero(" - H0 \t\t\t=", self.cosmo["H0"])
        self._print_zero(" - Omega_m \t\t=", self.cosmo["Omega_m"])
        self._print_zero(" - PowerSpecFile \t=", self.cosmo["PowerSpecFile"])
        self._check_exist(self.cosmo["PowerSpecFile"])
        self._print_zero(" - ScaleDepGrowth \t=", self.cosmo["ScaleDepGrowth"])
        self._print_zero(" - GrowthFile \t\t=", self.cosmo["GrowthFile"])
        self._check_exist(self.cosmo["GrowthFile"])

        # Read in Siminfo
        self._print_zero()
        self._print_zero(" Siminfo:")

        self.siminfo["Boxsize"] = float(params["Siminfo"]["Boxsize"])
        self.siminfo["Ngrid"] = int(params["Siminfo"]["Ngrid"])

        self._print_zero()
        self._print_zero(" - Boxsize \t\t=", self.siminfo["Boxsize"])
        self._print_zero(" - Ngrid \t\t=", self.siminfo["Ngrid"])

        # Read in Constraints
        if self._check_param_key(params, "Constraints"):

            self._print_zero()
            self._print_zero(" Constraints:")

            self.constraints["Fname"] = str(params["Constraints"]["Fname"])
            self.constraints["z_eff"] = float(params["Constraints"]["z_eff"])
            self.constraints["Rg"] = float(params["Constraints"]["Rg"])

            if self._check_param_key(params["Constraints"], "CorrFile"):
                self.constraints["CorrFile"] = params["Constraints"]["CorrFile"]
            if self._check_param_key(params["Constraints"], "CovFile"):
                self.constraints["CovFile"] = params["Constraints"]["CovFile"]
            self.constraints["Sigma_NL"] = float(params["Constraints"]["Sigma_NL"])

            self._print_zero()
            self._check_exist(self.constraints["Fname"])
            self._print_zero(" - Fname \t\t=", self.constraints["Fname"])
            self._print_zero(" - z_eff \t\t=", self.constraints["z_eff"])
            self._print_zero(" - Rg \t\t\t=", self.constraints["Rg"])

            if self.constraints["CorrFile"] is not None:
                 self._check_exist(self.constraints["CorrFile"])
            self._print_zero(" - CorrFile \t\t=", self.constraints["CorrFile"])
            if self.constraints["CovFile"] is not None:
                 self._check_exist(self.constraints["CovFile"])
            self._print_zero(" - CovFile \t\t=", self.constraints["CovFile"])

            self._print_zero(" - Sigma_NL \t\t=", self.constraints["Sigma_NL"])

            # Defines RZA Method -- for future use not currently implemented.
            if self._check_param_key(params["Constraints"], "RZA_Method"):
                self.constraints["RZA_Method"] = int(params["Constraints"]["RZA_Method"])
                self._print_zero(" - RZA_Method \t\t=", self.constraints["RZA_Method"])

            if self._check_param_key(params["Constraints"], "WF"):
                self.what2run["WF"] = bool(params["Constraints"]["WF"])
            else:
                self.what2run["WF"] = False

            if self._check_param_key(params["Constraints"], "RZA"):
                self.what2run["RZA"] = bool(params["Constraints"]["RZA"])
            else:
                self.what2run["RZA"] = False

            # RZA can only be performed in WF is computed
            if self.what2run["RZA"]:
                self.what2run["WF"] = True

            self._print_zero(" - WF \t\t\t=", io.bool2yesno(self.what2run["WF"]))
            self._print_zero(" - RZA \t\t\t=", io.bool2yesno(self.what2run["RZA"]))

        # ICs
        if self._check_param_key(params, "ICs"):

            self._print_zero()
            self._print_zero(" ICs:")

            if self._check_param_key(params["ICs"], "Seed"):
                self.ICs["Seed"] = int(params["ICs"]["Seed"])
                self._print_zero()
                self._print_zero(" - Seed \t\t=", self.ICs["Seed"])
            elif self._check_param_key(params["ICs"], "WNFile"):
                self.ICs["WNFile"] = str(params["ICs"]["WNFile"])
                self._print_zero()
                self._print_zero(" - WNFile \t\t=", self.ICs["WNFile"])
            else:
                self.ERROR = True
                self._print_zero(" ERROR: Must specify either a Seed or WNFile.")
                self._break4error()

            self.ICs["z_ic"] = float(params["ICs"]["z_ic"])
            self._print_zero(" - z_ic \t\t=", self.ICs["z_ic"])

            if self._check_param_key(params["ICs"], "CR"):
                self.what2run["CR"] = bool(params["ICs"]["CR"])
            else:
                self.what2run["CR"] = False

            self.what2run["IC"] = True

            self._print_zero(" - CR \t\t\t=", io.bool2yesno(self.what2run["CR"]))
            self._print_zero(" - IC \t\t\t=", io.bool2yesno(self.what2run["IC"]))

        else:
            self.what2run["IC"] = False

        # Outputs
        self._print_zero()
        self._print_zero(" Outputs:")

        self.outputs["OutputFolder"] = str(params["Outputs"]["OutputFolder"])
        self.outputs["Prefix"] = str(params["Outputs"]["Prefix"])

        self._print_zero()
        self._print_zero(" - OutputFolder \t=", self.outputs["OutputFolder"])
        self._print_zero(" - Prefix \t\t=", self.outputs["Prefix"])


    def read_paramfile(self, yaml_fname):
        """Reads parameter file."""
        self.params, self.ERROR = io.read_paramfile(yaml_fname, MPI=self.MPI)
        self._break4error()
        self._read_params(self.params)
        self._break4error()


    def _get_growth_D(self, redshift, kmag=None):
        """Returns the linear growth function from tabulated scale dependent and
        independent linear growth functions."""
        if self.cosmo["ScaleDepGrowth"]:
            return theory.get_growth_D(redshift, self.growth_z, self.growth_Dzk,
                kval=kmag, karray=self.growth_kh)
        else:
            return theory.get_growth_D(redshift, self.growth_z, self.growth_Dz)


    def _get_growth_f(self, redshift, kmag=None):
        """Returns the linear growth rate from tabulated scale dependent and
        independent linear growth rate."""
        if self.cosmo["ScaleDepGrowth"]:
            return theory.get_growth_f(redshift, self.growth_z, self.growth_fzk,
                kval=kmag, karray=self.growth_kh)
        else:
            return theory.get_growth_f(redshift, self.growth_z, self.growth_fz)

    def _prep_grid(self):
        self.SBX = fiesta.coords.MPI_SortByX(self.MPI)
        self.SBX.settings(self.siminfo["Boxsize"], self.siminfo["Ngrid"])
        self.SBX.limits4grid()


    def _prep_theory(self):
        self._print_zero()
        self._print_zero(" Theory")
        self._print_zero(" ======")
        self._print_zero()

        self._print_zero(" - Load PowerSpecFile :", self.cosmo["PowerSpecFile"])
        data = np.load(self.cosmo["PowerSpecFile"])
        self.theory_kh, self.theory_pk = data['kh'], data['pk']

        self._print_zero(" - Create P(k) interpolator")
        self.kmin, self.kmax = self.theory_kh.min(), self.theory_kh.max()

        self.interp_pk = interp1d(self.theory_kh, self.theory_pk, kind='cubic',
            bounds_error=False, fill_value=0.)

        self._print_zero()
        self._print_zero(" - Load GrowthFile :", self.cosmo["GrowthFile"])
        self._print_zero(" - ScaleDepGrowth :", self.cosmo["ScaleDepGrowth"])

        if self.cosmo["ScaleDepGrowth"]:
            data = np.load(self.cosmo["GrowthFile"])
            growth_z, growth_Hz, growth_kh, growth_Dzk, growth_fzk = data['z'], data['Hz'], data['kh'], data['Dzk'], data['fzk']
        else:
            data = np.load(self.cosmo["GrowthFile"])
            growth_z, growth_Hz, growth_Dz, growth_fz = data['z'], data['Hz'], data['Dz'], data['fz']

        self._print_zero()
        self._print_zero(" - Create H(z) interpolator")
        self.interp_Hz = interp1d(growth_z, growth_Hz/(self.cosmo["H0"]*1e-2), kind='cubic')

        self._print_zero()
        self._print_zero(" - Store D(z,k) and f(z,k) for interpolation")

        if self.cosmo["ScaleDepGrowth"]:
            self.growth_z = growth_z
            self.growth_kh = growth_kh
            self.growth_Dzk = growth_Dzk
            self.growth_fzk = growth_fzk
        else:
            self.growth_z = growth_z
            self.growth_Dz = growth_Dz
            self.growth_fz = growth_fz


    def _check_constraints(self):
        self._print_zero()
        self._print_zero(" - Prepare Constraints")
        # Normalise direction
        self._print_zero(" -- Normalize velocity unit vector")
        norm = np.sqrt(self.cons_ex**2. + self.cons_ey**2. + self.cons_ez**2.)
        self.cons_ex /= norm
        self.cons_ey /= norm
        self.cons_ez /= norm
        # Keep only positions inside the box, r <= halfboxsize
        self._print_zero(" -- Remove constrained points outside of the simulation box")
        cond = np.where((self.cons_x >= 0.) & (self.cons_x <= self.siminfo["Boxsize"]) &
                        (self.cons_y >= 0.) & (self.cons_y <= self.siminfo["Boxsize"]) &
                        (self.cons_z >= 0.) & (self.cons_z <= self.siminfo["Boxsize"]))[0]
        self._print_zero(" -- Retained %i constrained points from %i" % (len(cond), len(self.cons_x)))
        self.cons_x = self.cons_x[cond]
        self.cons_y = self.cons_y[cond]
        self.cons_z = self.cons_z[cond]
        self.cons_ex = self.cons_ex[cond]
        self.cons_ey = self.cons_ey[cond]
        self.cons_ez = self.cons_ez[cond]
        self.cons_c = self.cons_c[cond]
        self.cons_c_err = self.cons_c_err[cond]
        self.cons_c_type = self.cons_c_type[cond]
        if self.cons_c_RR is not None:
            self.cons_c_RR = self.cons_c_RR[cond]


    def _prep_constraints(self):
        self._print_zero()
        self._print_zero(" Constraints")
        self._print_zero(" ===========")
        self._print_zero()
        # Basic properties of the sims
        self.halfsize = self.siminfo["Boxsize"]/2.
        # Load constraints
        self._print_zero(" - Load constraint file :", self.constraints["Fname"])
        _x, _y, _z, _ex, _ey, _ez, _c, _c_err, _c_type = io.load_constraints(self.constraints["Fname"])
        self.cons_x, self.cons_y, self.cons_z = _x, _y, _z
        self.cons_ex, self.cons_ey, self.cons_ez = _ex, _ey, _ez
        self.cons_c, self.cons_c_err, self.cons_c_type = _c, _c_err, _c_type
        # Move position to the center of the box
        self._print_zero(" - Move constraints to the center of the simulation box")
        self.cons_x += self.halfsize
        self.cons_y += self.halfsize
        self.cons_z += self.halfsize
        self._check_constraints()


    def _save_correlators(self):
        if self.rank == 0:
            fname_prefix = self._get_fname_prefix()
            fname = fname_prefix + "correlator.npz"
            self._print_zero(" - Save correlation function as :", fname)
            io.save_correlators(fname, self.corr_redshift, self.corr_r, self.corr_xi,
                self.corr_zeta_p, self.corr_zeta_u, self.corr_psiR_pp, self.corr_psiT_pp,
                self.corr_psiR_pu, self.corr_psiT_pu, self.corr_psiR_uu, self.corr_psiT_uu,
                filetype='npz')


    def _load_correlators(self):
        if self.constraints["CorrFile"] is not None:
            self._print_zero(" - loading correlator file %s" % self.constraints["CorrFile"])
            if self.rank == 0:
                fname = self.constraints["CorrFile"]
                redshift, r, xi, zeta_p, zeta_u, psiR_pp, psiT_pp, psiR_pu, psiT_pu, psiR_uu, psiT_uu = io.load_correlators(fname)
            else:
                redshift = None
                r, xi, zeta_p, zeta_u = None, None, None, None
                psiR_pp, psiT_pp, psiR_pu, psiT_pu, psiR_uu, psiT_uu = None, None, None, None, None, None
            redshift = self.MPI.broadcast(redshift)
            if self.corr_redshift == redshift:
                self._print_zero(" - Correlation redshift %0.2f matches desired redshift %0.2f, storing correlators for use." % (redshift, self.corr_redshift))
                self.corr_r = self.MPI.broadcast(r)
                self.corr_xi = self.MPI.broadcast(xi)
                self.corr_zeta_p = self.MPI.broadcast(zeta_p)
                self.corr_zeta_u = self.MPI.broadcast(zeta_u)
                self.corr_psiR_pp = self.MPI.broadcast(psiR_pp)
                self.corr_psiT_pp = self.MPI.broadcast(psiT_pp)
                self.corr_psiR_pu = self.MPI.broadcast(psiR_pu)
                self.corr_psiT_pu = self.MPI.broadcast(psiT_pu)
                self.corr_psiR_uu = self.MPI.broadcast(psiR_uu)
                self.corr_psiT_uu = self.MPI.broadcast(psiT_uu)
                return True
            else:
                self._print_zero(" - Correlation redshift %0.2f does not matches desired redshift %0.2f, need to compute them." % (redshift, self.corr_redshift))
                return False
        else:
            return False


    def _calc_correlators(self):

        self._print_zero(" - Compute correlators in parallel")

        #self.sim_kmin = None
        #self.sim_kmax = None
        self.sim_kmin = shift.cart.get_kf(self.siminfo["Boxsize"])
        self.sim_kmax = np.sqrt(3.)*shift.cart.get_kn(self.siminfo["Boxsize"], self.siminfo["Ngrid"])

        self.corr_r = np.logspace(-2, np.log10(np.sqrt(3.)*self.siminfo["Boxsize"]), 100)

        dx = self.siminfo["Boxsize"]/self.siminfo["Ngrid"]

        Dz2 = self._get_growth_D(self.corr_redshift, kmag=self.theory_kh)**2.
        fz0 = self._get_growth_f(self.corr_redshift, kmag=self.theory_kh)

        _corr_r = self.MPI.split_array(self.corr_r)
        _Rg = self.constraints["Rg"]

        self._print_zero(" -- Computing xi(r)")

        _xi = theory.pk2xi(_corr_r, self.theory_kh, Dz2*self.theory_pk, kmin=self.sim_kmin,
            kmax=self.sim_kmax, kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6), Rg=_Rg)

        self._print_zero(" -- Computing zeta^p(r)")

        _zeta_p = theory.pk2zeta(_corr_r, self.theory_kh, Dz2*self.theory_pk, fk=None,
            kmin=self.sim_kmin, kmax=self.sim_kmax, kfactor=100, kbinsmin=int(1e4),
            kbinsmax=int(1e6), Rg=_Rg)

        self._print_zero(" -- Computing zeta^u(r)")

        if self.cosmo["ScaleDepGrowth"]:
            _zeta_u = theory.pk2zeta(_corr_r, self.theory_kh, Dz2*self.theory_pk, fk=fz0, kmin=self.sim_kmin, kmax=self.sim_kmax,
                kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6), Rg=_Rg)
        else:
            _zeta_u = fz0*np.copy(_zeta_p)

        self._print_zero(" -- Computing psiR^pp(r) and psiT^pp(r)")

        _psiR_pp = theory.pk2psiR(_corr_r, self.theory_kh, Dz2*self.theory_pk, fk=None, kmin=self.sim_kmin, kmax=self.sim_kmax,
            kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6), Rg=_Rg)
        _psiT_pp = theory.pk2psiT(_corr_r, self.theory_kh, Dz2*self.theory_pk, fk=None, kmin=self.sim_kmin, kmax=self.sim_kmax,
            kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6), Rg=_Rg)

        self._print_zero(" -- Computing psiR^pu(r) and psiT^pu(r)")

        if self.cosmo["ScaleDepGrowth"]:
            _psiR_pu = theory.pk2psiR(_corr_r, self.theory_kh, Dz2*self.theory_pk, fk=np.sqrt(fz0), kmin=self.sim_kmin, kmax=self.sim_kmax,
                kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6), Rg=_Rg)
            _psiT_pu = theory.pk2psiT(_corr_r, self.theory_kh, Dz2*self.theory_pk, fk=np.sqrt(fz0), kmin=self.sim_kmin, kmax=self.sim_kmax,
                kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6), Rg=_Rg)
        else:
            _psiR_pu = fz0*np.copy(_psiR_pp)
            _psiT_pu = fz0*np.copy(_psiT_pp)

        self._print_zero(" -- Computing psiR^uu(r) and psiT^uu(r)")

        if self.cosmo["ScaleDepGrowth"]:
            _psiR_uu = theory.pk2psiR(_corr_r, self.theory_kh, Dz2*self.theory_pk, fk=fz0, kmin=self.sim_kmin, kmax=self.sim_kmax,
                kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6), Rg=_Rg)
            _psiT_uu = theory.pk2psiT(_corr_r, self.theory_kh, Dz2*self.theory_pk, fk=fz0, kmin=self.sim_kmin, kmax=self.sim_kmax,
                kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6), Rg=_Rg)
        else:
            _psiR_uu = (fz0**2)*np.copy(_psiR_pp)
            _psiT_uu = (fz0**2)*np.copy(_psiT_pp)

        _psiR_uu = theory.pk2psiR(_corr_r, self.theory_kh, Dz2*self.theory_pk, fk=fz0, kmin=self.sim_kmin, kmax=self.sim_kmax,
            kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6), Rg=_Rg)
        _psiT_uu = theory.pk2psiT(_corr_r, self.theory_kh, Dz2*self.theory_pk, fk=fz0, kmin=self.sim_kmin, kmax=self.sim_kmax,
            kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6), Rg=_Rg)

        self.MPI.wait()

        self._print_zero()
        self._print_zero(" - Collect correlation functions")

        _corr_r = self.MPI.collect(_corr_r)
        _xi = self.MPI.collect(_xi)
        _zeta_p = self.MPI.collect(_zeta_p)
        _zeta_u = self.MPI.collect(_zeta_u)
        _psiR_pp = self.MPI.collect(_psiR_pp)
        _psiT_pp = self.MPI.collect(_psiT_pp)
        _psiR_pu = self.MPI.collect(_psiR_pu)
        _psiT_pu = self.MPI.collect(_psiT_pu)
        _psiR_uu = self.MPI.collect(_psiR_uu)
        _psiT_uu = self.MPI.collect(_psiT_uu)

        self._print_zero(" - Broadcast correlation functions")

        _corr_r = self.MPI.broadcast(_corr_r)
        _xi = self.MPI.broadcast(_xi)
        _zeta_p = self.MPI.broadcast(_zeta_p)
        _zeta_u = self.MPI.broadcast(_zeta_u)
        _psiR_pp = self.MPI.broadcast(_psiR_pp)
        _psiT_pp = self.MPI.broadcast(_psiT_pp)
        _psiR_pu = self.MPI.broadcast(_psiR_pu)
        _psiT_pu = self.MPI.broadcast(_psiT_pu)
        _psiR_uu = self.MPI.broadcast(_psiR_uu)
        _psiT_uu = self.MPI.broadcast(_psiT_uu)

        self.corr_r = np.concatenate([np.array([0.]), _corr_r])
        self.corr_xi = np.concatenate([np.array([_xi[0]]), _xi])
        self.corr_zeta_p = np.concatenate([np.array([0.]), _zeta_p])
        self.corr_zeta_u = np.concatenate([np.array([0.]), _zeta_u])
        self.corr_psiR_pp = np.concatenate([np.array([_psiR_pp[0]]), _psiR_pp])
        self.corr_psiT_pp = np.concatenate([np.array([_psiT_pp[0]]), _psiT_pp])
        self.corr_psiR_pu = np.concatenate([np.array([_psiR_pu[0]]), _psiR_pu])
        self.corr_psiT_pu = np.concatenate([np.array([_psiT_pu[0]]), _psiT_pu])
        self.corr_psiR_uu = np.concatenate([np.array([_psiR_uu[0]]), _psiR_uu])
        self.corr_psiT_uu = np.concatenate([np.array([_psiT_uu[0]]), _psiT_uu])


    def _prep_correlators(self, redshift):
        self._print_zero()
        self._print_zero(" Correlators")
        self._print_zero(" ===========")
        self._print_zero()

        self.corr_redshift = redshift

        if self._load_correlators() == False:
            self._calc_correlators()
            self._save_correlators()

        self._print_zero(" - Construct interpolators")

        self.interp_xi = interp1d(self.corr_r, self.corr_xi, kind='cubic', bounds_error=False, fill_value=0.)
        self.interp_zeta_p = interp1d(self.corr_r, self.corr_zeta_p, kind='cubic', bounds_error=False, fill_value=0.)
        self.interp_zeta_u = interp1d(self.corr_r, self.corr_zeta_u, kind='cubic', bounds_error=False, fill_value=0.)
        self.interp_psiR_pp = interp1d(self.corr_r, self.corr_psiR_pp, kind='cubic', bounds_error=False, fill_value=0.)
        self.interp_psiT_pp = interp1d(self.corr_r, self.corr_psiT_pp, kind='cubic', bounds_error=False, fill_value=0.)
        self.interp_psiR_pu = interp1d(self.corr_r, self.corr_psiR_pu, kind='cubic', bounds_error=False, fill_value=0.)
        self.interp_psiT_pu = interp1d(self.corr_r, self.corr_psiT_pu, kind='cubic', bounds_error=False, fill_value=0.)
        self.interp_psiR_uu = interp1d(self.corr_r, self.corr_psiR_uu, kind='cubic', bounds_error=False, fill_value=0.)
        self.interp_psiT_uu = interp1d(self.corr_r, self.corr_psiT_uu, kind='cubic', bounds_error=False, fill_value=0.)


    def _save_cov(self):
        fname = self._get_fname_prefix() + 'cov.npz'
        np.savez(fname, cov=self.cov, sigma_NL=self.constraints["Sigma_NL"], c=self.cons_c, type_c=self.cons_c_type)


    def compute_eta(self):
        self._print_zero()
        self._print_zero(" Compute eta-vector")
        self._print_zero(" ==================")
        self._print_zero()

        x1, x2 = self.MPI.create_split_ndgrid([self.cons_x, self.cons_x], [False, True])
        y1, y2 = self.MPI.create_split_ndgrid([self.cons_y, self.cons_y], [False, True])
        z1, z2 = self.MPI.create_split_ndgrid([self.cons_z, self.cons_z], [False, True])

        ex1, ex2 = self.MPI.create_split_ndgrid([self.cons_ex, self.cons_ex], [False, True])
        ey1, ey2 = self.MPI.create_split_ndgrid([self.cons_ey, self.cons_ey], [False, True])
        ez1, ez2 = self.MPI.create_split_ndgrid([self.cons_ez, self.cons_ez], [False, True])

        type1, type2 = self.MPI.create_split_ndgrid([self.cons_c_type, self.cons_c_type], [False, True])

        self._print_zero(" - Compute vel-vel covariance matrix in parallel")

        cov_cc = theory.get_cc_matrix_fast(x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
            type1, type2, self.corr_redshift, self.interp_Hz, self.interp_xi, self.interp_zeta_p,
            self.interp_zeta_u, self.interp_psiR_pp, self.interp_psiT_pp, self.interp_psiR_pu,
            self.interp_psiT_pu, self.interp_psiR_uu, self.interp_psiT_uu, self.siminfo["Boxsize"],
            minlogr=-2)

        self._print_zero(" - Collect vel-vel covariance matrix [at MPI.rank = 0]")

        self.cov = self.MPI.collect(cov_cc)
        self.cov = self.MPI.broadcast(self.cov)

        if self.rank == 0:
            self.cov += np.diag(self.cons_c_err**2. + self.constraints["Sigma_NL"]**2.)

            self._save_cov()

            self._print_zero(" - Inverting matrix [at MPI.rank = 0]")
            self.inv = np.linalg.inv(self.cov)

            self._print_zero(" - Compute eta vector [at MPI.rank = 0]")
            self.eta = self.inv.dot(self.cons_c)

        self.MPI.wait()

        self._print_zero(" - Broadcast eta vector")
        self.eta = self.MPI.broadcast(self.eta)
        self.inv = self.MPI.broadcast(self.inv)

        self.MPI.wait()


    def prep(self):
        """Runs all the grid, theory and constraint preparation functions."""
        self._prep_grid()
        self._prep_theory()
        self._prep_constraints()
        self._prep_correlators(self.constraints["z_eff"])


    def get_grid3D(self):
        if self.x3D is None:
            self._print_zero(" - Construct cartesian grid")
            self.x3D, self.y3D, self.z3D = shift.cart.mpi_grid3D(self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)
            self.x_shape = np.shape(self.x3D)


    def flatten_grid3D(self):
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
        if self.kx3D is None:
            self._print_zero(" - Construct Fourier grid")
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
        cov_du = self.interp_zeta_u(r)
        du = - adot*cov_du*norm_rx*self.cons_ex - adot*cov_du*norm_ry*self.cons_ey - adot*cov_du*norm_rz*self.cons_ez
        if self.rank == 0:
            utils.progress_bar(i, total, explanation=" ---- ", indexing=True)
        return du.dot(self.eta)


    def _MPI_save_xyz(self, suffix="XYZ"):
        fname_prefix = self._get_fname_prefix()
        fname = fname_prefix + suffix + "_" + str(self.rank) + ".npz"
        if io.isfile(fname) is False:
            check = True
        else:
            data = np.load(fname)
            if data['Boxsize'] == self.siminfo["Boxsize"] and data["Ngrid"] == self.siminfo["Ngrid"]:
                check = False
            else:
                check = True
        if check:
            self._print_zero(" - Save XYZ :", fname_prefix+suffix+"_[0-%i].npz" % (self.MPI.size-1))
            np.savez(fname, Boxsize=self.siminfo["Boxsize"], Ngrid=self.siminfo["Ngrid"],
                x3D=self.x3D, y3D=self.y3D, z3D=self.z3D)


    def _MPI_savez(self, suffix, **kwarg):
        fname_prefix = self._get_fname_prefix()
        fname = fname_prefix + suffix + "_" + str(self.rank) + ".npz"
        self._print_zero(" - Saving to :", fname_prefix+suffix+"_[0-%i].npz" % (self.MPI.size-1))
        np.savez(fname, **kwarg)


    def save_WF(self, dens_WF):
        self._MPI_save_xyz()
        suffix = "WF"
        self._MPI_savez(suffix, dens=dens_WF)


    def save_err_WF(self, dens_err_WF):
        self._MPI_save_xyz()
        suffix = "WF_err"
        self._MPI_savez(suffix, err=dens_err_WF)


    def get_WF(self):
        self._print_zero()
        self._print_zero(" Compute Wiener Filter")
        self._print_zero(" =====================")
        self._print_zero()

        self.get_grid3D()
        self.flatten_grid3D()

        self._print_zero()
        self._print_zero(" - Computing Wiener Filter density")

        prefix = " ---- "

        typei = 0
        exi, eyi, ezi = 1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)

        dens_WF = theory.get_corr_dot_eta_fast(self.x3D, self.cons_x, self.y3D, self.cons_y,
            self.z3D, self.cons_z, exi, self.cons_ex, eyi, self.cons_ey, ezi, self.cons_ez,
            typei, self.cons_c_type, self.corr_redshift, self.interp_Hz, self.interp_xi,
            self.interp_zeta_p, self.interp_zeta_u, self.interp_psiR_pp, self.interp_psiT_pp,
            self.interp_psiR_pu, self.interp_psiT_pu, self.interp_psiR_uu, self.interp_psiT_uu,
            self.eta, self.siminfo["Boxsize"], self._lenpro+2, len(prefix), prefix,
            mpi_rank=self.MPI.rank, minlogr=-2)

        # self._print_zero()
        # self._print_zero(" - Computing Wiener Filter density error")
        #
        # dens_err_WF = theory.get_cc_float_fast(0., 0., 0., 0., 0., 0., exi, exi, exi, exi,
        #     exi, exi, 0, 0, self.corr_redshift, self.interp_Hz, self.interp_xi, self.interp_zeta_p,
        #     self.interp_zeta_u, self.interp_psiR_pp, self.interp_psiT_pp, self.interp_psiR_pu,
        #     self.interp_psiT_pu, self.interp_psiR_uu, self.interp_psiT_uu, self.siminfo["Boxsize"],
        #     minlogr=-2)
        #
        # dens_err_WF -= theory.get_corr1_dot_inv_dot_corr2_fast(self.x3D, np.copy(self.x3D), self.cons_x,
        #     self.y3D, np.copy(self.y3D), self.cons_y, self.z3D, np.copy(self.z3D), self.cons_z,
        #     exi, self.cons_ex, eyi, self.cons_ey, ezi, self.cons_ez, typei, typei, self.cons_c_type,
        #     self.corr_redshift, self.interp_Hz, self.interp_xi, self.interp_zeta_p, self.interp_zeta_u,
        #     self.interp_psiR_pp, self.interp_psiT_pp, self.interp_psiR_pu, self.interp_psiT_pu,
        #     self.interp_psiR_uu, self.interp_psiT_uu, self.inv, self.siminfo["Boxsize"], self._lenpro+2,
        #     len(prefix), prefix, mpi_rank=self.MPI.rank, minlogr=-2, nlogr=1000)

        self.unflatten_grid3D()
        dens_WF = dens_WF.reshape(self.x_shape)
        #dens_err_WF = dens_err_WF.reshape(self.x_shape)

        self._print_zero()
        self.save_WF(dens_WF)
        #self.save_err_WF(dens_err_WF)

        if self.what2run["RZA"]:
            self.dens_WF = dens_WF


    def start_FFT(self, Ngrid):
        if self.FFT is None or self.FFT_Ngrid != Ngrid:
            self.FFT_Ngrid = Ngrid
            Ngrids = np.array([Ngrid, Ngrid, Ngrid], dtype=int)
            self.FFT = self.MPI.mpi_fft_start(Ngrids)


    def complex_zeros(self, shape):
        return np.zeros(shape) + 1j*np.zeros(shape)


    def dens2psi(self, dens):
        self.start_FFT(self.siminfo["Ngrid"])
        self.get_grid3D()
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
        z0 = redshift
        Hz = self.interp_Hz(z0)

        adot = theory.z2a(z0)*Hz

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
            vel_kx[cond] *= adot*fk
            vel_ky[cond] *= adot*fk
            vel_kz[cond] *= adot*fk

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


    def _add_buffer_in_x(self, f):
        f_send_down = self.MPI.send_down(f[0])
        f_send_up = self.MPI.send_up(f[-1])
        fnew = np.concatenate([np.array([f_send_up]), f, np.array([f_send_down])])
        return fnew


    def _get_buffer_range(self):
        dx = self.siminfo["Boxsize"]/self.siminfo["Ngrid"]
        xmin = np.min(self.x3D) - dx/2. - dx
        xmax = np.max(self.x3D) + dx/2. + dx
        return xmin, xmax


    def _unbuffer_in_x(self, f):
        fnew = f[1:-1]
        return fnew


    def get_RZA(self):
        self._print_zero()
        self._print_zero(" Apply Reverse Zeldovich Approximation")
        self._print_zero(" =====================================")
        self._print_zero()

        self._print_zero(" - Converting WF density to displacement Psi")
        self._print_zero()

        psi_x, psi_y, psi_z = self.dens2psi(self.dens_WF)

        self._print_zero(" - Add buffer region to Psi for interpolation")
        psi_x = self._add_buffer_in_x(psi_x)
        psi_y = self._add_buffer_in_x(psi_y)
        psi_z = self._add_buffer_in_x(psi_z)
        xmin, xmax = self._get_buffer_range()

        if self.rank == 0:
            data = np.column_stack([self.cons_x, self.cons_y, self.cons_z, self.cons_ex,
                self.cons_ey, self.cons_ez, self.cons_c, self.cons_c_err])
        else:
            data = None

        self.SBX.input(data)
        data = self.SBX.distribute()
        cons_x, cons_y, cons_z, cons_ex, cons_ey, cons_ez, cons_c, cons_c_err = \
            data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5], data[:,6], data[:,7]

        self._print_zero(" - Interpolating displacement Psi at constraint positions")

        if len(cons_x) != 0:
            cons_psi_x = fiesta.interp.trilinear(psi_x, [xmax-xmin, self.siminfo["Boxsize"], self.siminfo["Boxsize"]],
                cons_x, cons_y, cons_z, origin=[xmin, 0., 0.], periodic=[False, True, True])
            cons_psi_y = fiesta.interp.trilinear(psi_y, [xmax-xmin, self.siminfo["Boxsize"], self.siminfo["Boxsize"]],
                cons_x, cons_y, cons_z, origin=[xmin, 0., 0.], periodic=[False, True, True])
            cons_psi_z = fiesta.interp.trilinear(psi_z, [xmax-xmin, self.siminfo["Boxsize"], self.siminfo["Boxsize"]],
                cons_x, cons_y, cons_z, origin=[xmin, 0., 0.], periodic=[False, True, True])
            # x_shape = self.x_shape
            # dx = (xmax-xmin)/(x_shape[0]+2)
            # ix = np.floor((cons_x - xmin)/dx).astype('int')
            # dy = (self.siminfo["Boxsize"])/x_shape[1]
            # iy = np.floor((cons_y)/dy).astype('int')
            # dz = (self.siminfo["Boxsize"])/x_shape[2]
            # iz = np.floor((cons_z)/dz).astype('int')
            # cons_psi_x = psi_x[ix,iy,iz]
            # cons_psi_y = psi_y[ix,iy,iz]
            # cons_psi_z = psi_z[ix,iy,iz]
        else:
            cons_psi_x, cons_psi_y, cons_psi_z = None, None, None

        self.MPI.wait()

        self._print_zero(" - Remove buffer region to Psi for interpolation")
        self._print_zero()

        psi_x = self._unbuffer_in_x(psi_x)
        psi_y = self._unbuffer_in_x(psi_y)
        psi_z = self._unbuffer_in_x(psi_z)

        self._print_zero(" - Applying RZA")
        self._print_zero()

        if self.constraints["RZA_Method"] == 2:
            # This assumes Method II of https://theses.hal.science/tel-01127294/document see page 121
            cons_rza_x = cons_x - cons_psi_x
            cons_rza_y = cons_y - cons_psi_y
            cons_rza_z = cons_z - cons_psi_z
            cons_rza_ex = np.copy(cons_ex)
            cons_rza_ey = np.copy(cons_ey)
            cons_rza_ez = np.copy(cons_ez)
            cons_rza_c = np.copy(cons_c)
            cons_rza_c_err = np.copy(cons_c_err)

        # elif self.constraints["RZA_Method"] == 3:
        #     # See above paper for Method III
        #     cons_rza_x = cons_x - cons_psi_x
        #     cons_rza_y = cons_y - cons_psi_y
        #     cons_rza_z = cons_z - cons_psi_z
        #     cons_rza_ex = np.copy(cons_ex)
        #     cons_rza_ey = np.copy(cons_ey)
        #     cons_rza_ez = np.copy(cons_ez)
        #     cons_rza_c = np.copy(cons_c)
        #     cons_rza_c_err = np.zeros(len(cons_c_err))
        #     cons_rza_c = np.sqrt(cons_rza_x**2. + cons_rza_y**2. + cons_rza_z**2.)
        #     cons_rza_ex = cons_rza_x / cons_rza_c
        #     cons_rza_ey = cons_rza_y / cons_rza_c
        #     cons_rza_ez = cons_rza_z / cons_rza_c

        self.cons_x = self.MPI.collect_noNone(cons_rza_x)
        self.cons_y = self.MPI.collect_noNone(cons_rza_y)
        self.cons_z = self.MPI.collect_noNone(cons_rza_z)

        self.cons_ex = self.MPI.collect_noNone(cons_rza_ex)
        self.cons_ey = self.MPI.collect_noNone(cons_rza_ey)
        self.cons_ez = self.MPI.collect_noNone(cons_rza_ez)

        self.cons_c = self.MPI.collect_noNone(cons_rza_c)
        self.cons_c_err = self.MPI.collect_noNone(cons_rza_c_err)

        self.cons_x = self.MPI.broadcast(self.cons_x)
        self.cons_y = self.MPI.broadcast(self.cons_y)
        self.cons_z = self.MPI.broadcast(self.cons_z)

        self.cons_ex = self.MPI.broadcast(self.cons_ex)
        self.cons_ey = self.MPI.broadcast(self.cons_ey)
        self.cons_ez = self.MPI.broadcast(self.cons_ez)

        self.cons_c = self.MPI.broadcast(self.cons_c)
        self.cons_c_err = self.MPI.broadcast(self.cons_c_err)

        fname = self._get_fname_prefix() + 'rza.npz'
        self._print_zero(" - Saving RZA constraints to: %s" % fname)

        if self.rank == 0:
            np.savez(fname, x=self.cons_x-self.halfsize, y=self.cons_y-self.halfsize,
                z=self.cons_z-self.halfsize, ex=self.cons_ex, ey=self.cons_ey,
                ez=self.cons_ez, c=self.cons_c, c_err=self.cons_c_err)


    def save_WN(self, WN):
        self._MPI_save_xyz()
        suffix = "WN"
        self._MPI_savez(suffix, WN=WN)


    def save_dens(self, suffix):
        self._MPI_save_xyz()
        self._MPI_savez(suffix, dens=self.dens)


    def get_RR(self):
        self._print_zero()
        self._print_zero(" Construct Random Realisation")
        self._print_zero(" ============================")
        self._print_zero()

        self.get_grid3D()
        self.get_kgrid3D()
        kmag = self.get_kgrid_mag()

        self.start_FFT(self.siminfo["Ngrid"])

        if self.ICs["Seed"] is not None:
            seed = self.ICs["Seed"] + self.rank
            self._print_zero(" - Construct white noise field with seed %i" % seed)
            WN = field.get_white_noise(seed, *self.x_shape)
        elif self.ICs["WNFile"] is not None:
            fname = self.ICs["WNFile"]
            self._check_exist(fname)
            if self.rank == 0:
                data = np.load(fname)
                WN = data["WN"]
                if len(WN) != self.siminfo["Ngrid"]:
                    self.ERROR = True
                    self._print_zero(" ERROR: Ngrid = %i for WN does not match siminfo Ngrid = %i." % (len(WN), self.siminfo["Ngrid"]))
                    self._break4error()
                _x3D, _y3D, _z3D = shift.cart.grid3D(self.siminfo["Boxsize"], self.siminfo["Ngrid"])
            else:
                _x3D, _y3D, _z3D, WN = None, None, None, None
            self.MPI.wait()
            WN = self.SBX.distribute_grid3D(_x3D, _y3D, _z3D, WN)

        self.save_WN(WN)

        self._print_zero(" - FFT white noise field")

        WN_k = shift.cart.mpi_fft3D(WN, self.x_shape, self.siminfo["Boxsize"],
            self.siminfo["Ngrid"], self.FFT)

        self._print_zero(" - Colour white noise field to get density field")

        dx = self.siminfo["Boxsize"]/self.siminfo["Ngrid"]
        dens_RR_k = field.color_white_noise(WN_k, dx, kmag, self.interp_pk, mode='3D')

        self._print_zero(" - iFFT density field")

        self.dens = shift.cart.mpi_ifft3D(dens_RR_k, self.x_shape, self.siminfo["Boxsize"],
            self.siminfo["Ngrid"], self.FFT)

        self.save_dens("RR")


    def dens_at_z(self, dens, redshift, redshift_current=0.):
        z0 = redshift
        z1 = redshift_current
        if self.cosmo["ScaleDepGrowth"]:
            self.start_FFT(self.siminfo["Ngrid"])
            self.get_kgrid3D()
            kmag = self.get_kgrid_mag()
            dens_k = shift.cart.mpi_fft3D(dens, self.x_shape, self.siminfo["Boxsize"],
                self.siminfo["Ngrid"], self.FFT)
            cond = np.where(kmag != 0.)
            Dk = self._get_growth_D(z0, kmag=kmag[cond])
            Dk0 = self._get_growth_D(z1, kmag=kmag[cond])
            dens_k[cond] = (Dk/Dk0)*dens_k[cond]
            densz = shift.cart.mpi_ifft3D(dens_k, self.x_shape, self.siminfo["Boxsize"],
                self.siminfo["Ngrid"], self.FFT)
        else:
            Dz = self._get_growth_D(z0)
            Dz0 = self._get_growth_D(z1)
            densz = (Dz/Dz0)*dens
        return densz


    def compute_eta_CR(self):
        self._print_zero()
        self._print_zero(" Compute eta_CR-vector")
        self._print_zero(" =====================")
        self._print_zero()

        x1, x2 = self.MPI.create_split_ndgrid([self.cons_x, self.cons_x], [False, True])
        y1, y2 = self.MPI.create_split_ndgrid([self.cons_y, self.cons_y], [False, True])
        z1, z2 = self.MPI.create_split_ndgrid([self.cons_z, self.cons_z], [False, True])

        ex1, ex2 = self.MPI.create_split_ndgrid([self.cons_ex, self.cons_ex], [False, True])
        ey1, ey2 = self.MPI.create_split_ndgrid([self.cons_ey, self.cons_ey], [False, True])
        ez1, ez2 = self.MPI.create_split_ndgrid([self.cons_ez, self.cons_ez], [False, True])

        type1, type2 = self.MPI.create_split_ndgrid([self.cons_c_type, self.cons_c_type], [False, True])

        self._print_zero(" - Compute vel-vel covariance matrix in parallel")

        cov_cc = theory.get_cc_matrix_fast(x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
            type1, type2, self.corr_redshift, self.interp_Hz, self.interp_xi, self.interp_zeta_p,
            self.interp_zeta_u, self.interp_psiR_pp, self.interp_psiT_pp, self.interp_psiR_pu,
            self.interp_psiT_pu, self.interp_psiR_uu, self.interp_psiT_uu, self.siminfo["Boxsize"],
            minlogr=-2)

        self._print_zero(" - Collect vel-vel covariance matrix [at MPI.rank = 0]")

        cov_cc = self.MPI.collect(cov_cc)

        if self.rank == 0:
            cov_cc += np.diag(self.cons_c_err**2.)
            # add sigma_NL more error?

            self._print_zero(" - Inverting matrix [at MPI.rank = 0]")
            inv_cc = np.linalg.inv(cov_cc)

            self._print_zero(" - Compute eta_CR vector [at MPI.rank = 0]")
            self.eta_CR = inv_cc.dot(self.cons_c - self.cons_c_RR)

        self.MPI.wait()

        self._print_zero(" - Broadcast eta_CR vector")
        self.eta_CR = self.MPI.broadcast(self.eta_CR)


    def _get_CR_single(self, x, y, z, adot, i, total):
        rx = self.cons_x - x
        ry = self.cons_y - y
        rz = self.cons_z - z
        r = np.sqrt(rx**2. + ry**2. + rz**2.)
        norm_rx = np.copy(rx)/r
        norm_ry = np.copy(ry)/r
        norm_rz = np.copy(rz)/r
        cov_du = self.interp_zeta_u(r)
        du = - adot*cov_du*norm_rx*self.cons_ex - adot*cov_du*norm_ry*self.cons_ey - adot*cov_du*norm_rz*self.cons_ez
        if self.rank == 0:
            utils.progress_bar(i, total, explanation=" ---- ", indexing=True)
        return du.dot(self.eta_CR)


    def prep_CR(self):
        self._print_zero()
        self._print_zero(" Prepare for Constrained Realisation")
        self._print_zero(" ===================================")
        self._print_zero()

        z0 = self.constraints["z_eff"]

        self._print_zero(" - Scaling density to the z_eff=%0.2f of constraints" % self.constraints["z_eff"])

        self.dens = self.dens_at_z(self.dens, z0)

        self._print_zero(" - Computing displacement field Psi from density")

        psi_x, psi_y, psi_z = self.dens2psi(self.dens)

        self._print_zero(" - Computing velocity field from displacement field Psi")

        vel_x, vel_y, vel_z = self.psi2vel(z0, psi_x, psi_y, psi_z)

        del psi_x
        del psi_y
        del psi_z

        vel_x = self._add_buffer_in_x(vel_x)
        vel_y = self._add_buffer_in_x(vel_y)
        vel_z = self._add_buffer_in_x(vel_z)

        xmin, xmax = self._get_buffer_range()

        if self.rank == 0:
            data = np.column_stack([self.cons_x, self.cons_y, self.cons_z, self.cons_ex,
                self.cons_ey, self.cons_ez, self.cons_c, self.cons_c_err])
        else:
            data = None

        self.SBX.input(data)
        data = self.SBX.distribute()
        cons_x, cons_y, cons_z, cons_ex, cons_ey, cons_ez, cons_c, cons_c_err = \
            data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5], data[:,6], data[:,7]

        self._print_zero(" - Interpolating velocity at constraint positions")

        if len(cons_x) != 0:
            cons_vel_x = fiesta.interp.trilinear(vel_x, [xmax-xmin, self.siminfo["Boxsize"], self.siminfo["Boxsize"]],
                cons_x, cons_y, cons_z, origin=[xmin, 0., 0.], periodic=[False, True, True])
            cons_vel_y = fiesta.interp.trilinear(vel_y, [xmax-xmin, self.siminfo["Boxsize"], self.siminfo["Boxsize"]],
                cons_x, cons_y, cons_z, origin=[xmin, 0., 0.], periodic=[False, True, True])
            cons_vel_z = fiesta.interp.trilinear(vel_z, [xmax-xmin, self.siminfo["Boxsize"], self.siminfo["Boxsize"]],
                cons_x, cons_y, cons_z, origin=[xmin, 0., 0.], periodic=[False, True, True])
            cons_c_RR = cons_vel_x*cons_ex + cons_vel_y*cons_ey + cons_vel_z*cons_ez
            cons_data = np.column_stack([cons_x, cons_y, cons_z, cons_ex, cons_ey, cons_ez, cons_c, cons_c_err, cons_c_RR])
        else:
            cons_data = None

        self.MPI.wait()

        del vel_x
        del vel_y
        del vel_z

        cons_data = self.MPI.collect_noNone(cons_data)
        self.MPI.wait()

        cons_data = self.MPI.broadcast(cons_data)
        self.MPI.wait()

        self.cons_x, self.cons_y, self.cons_z = cons_data[:,0], cons_data[:,1], cons_data[:,2]
        self.cons_ex, self.cons_ey, self.cons_ez = cons_data[:,3], cons_data[:,4], cons_data[:,5]
        self.cons_c, self.cons_c_err, self.cons_c_RR = cons_data[:,6], cons_data[:,7], cons_data[:,8]

        self._check_constraints()

        self.MPI.wait()

        self.compute_eta_CR()


    def get_CR(self):
        self._print_zero()
        self._print_zero(" Construct Constrained Realisation")
        self._print_zero(" =================================")
        self._print_zero()

        z0 = self.constraints["z_eff"]

        self.get_grid3D()
        self.flatten_grid3D()

        self.dens = self.dens.flatten()

        # Hz = self.interp_Hz(z0)
        #
        # if self.constraints["Type"] == "Vel":
        #     adot = theory.z2a(z0)*Hz
        # elif self.constraints["Type"] == "Psi":
        #     adot = 1.

        self._print_zero(" - Computing Constrained Realisation density")

        prefix = " ---- "

        typei = 0
        exi, eyi, ezi = 1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)

        self.dens += theory.get_corr_dot_eta_fast(self.x3D, self.cons_x, self.y3D, self.cons_y,
            self.z3D, self.cons_z, exi, self.cons_ex, eyi, self.cons_ey, ezi, self.cons_ez,
            typei, self.cons_c_type, self.corr_redshift, self.interp_Hz, self.interp_xi,
            self.interp_zeta_p, self.interp_zeta_u, self.interp_psiR_pp, self.interp_psiT_pp,
            self.interp_psiR_pu, self.interp_psiT_pu, self.interp_psiR_uu, self.interp_psiT_uu,
            self.eta_CR, self.siminfo["Boxsize"], lenpro=self._lenpro+2, lenpre=len(prefix),
            prefix=prefix, mpi_rank=self.MPI.rank, minlogr=-2)

        self.unflatten_grid3D()
        self.dens = self.dens.reshape(self.x_shape)

        self._print_zero()
        self.save_dens("CR")

    def get_particle_mass(self):
        G_const = 6.6743e-11
        part_mass = 3.*self.cosmo["Omega_m"]*self.siminfo["Boxsize"]**3.
        part_mass /= 8.*np.pi*G_const*self.siminfo["Ngrid"]**3.
        part_mass *= 3.0857e2/1.9891
        part_mass /= 1e10
        return part_mass


    def get_IC(self):
        self._print_zero()
        self._print_zero(" Generate Initial Conditions")
        self._print_zero(" ===========================")
        self._print_zero()

        z0 = self.ICs["z_ic"]
        if self.what2run["CR"]:
            z1 = self.constraints["z_eff"]
        else:
            z1 = 0.

        self._print_zero(" - Scaling density from redshift %0.2f to %0.2f" % (z1, z0))
        self.dens = self.dens_at_z(self.dens, z0, redshift_current=z1)

        self.save_dens("IC")

        self._print_zero()
        self._print_zero(" - Computing IC positions and velocities using 1LPT")

        psi_x, psi_y, psi_z = self.dens2psi(self.dens)
        vel_x, vel_y, vel_z = self.psi2vel(z0, psi_x, psi_y, psi_z)

        pos_x = self.x3D + psi_x
        pos_y = self.y3D + psi_y
        pos_z = self.z3D + psi_z

        pos_x = pos_x.flatten()
        pos_y = pos_y.flatten()
        pos_z = pos_z.flatten()

        vel_x = vel_x.flatten()
        vel_y = vel_y.flatten()
        vel_z = vel_z.flatten()

        part_len = np.array([len(pos_x)])
        part_lens = self.MPI.collect(part_len)

        if self.rank == 0:
            part_id_offsets = np.cumsum(part_lens)
            self.MPI.send(part_id_offsets, tag=11)
        else:
            part_id_offsets = self.MPI.recv(0, tag=11)

        self.MPI.wait()

        part_id_offsets = np.array([0] + np.ndarray.tolist(part_id_offsets))

        part_mass = self.get_particle_mass()

        self._print_zero(" - Particle mass = %0.6f" % part_mass)
        self.MPI.wait()

        header = {
          'nfiles'        : self.MPI.size,
          'massarr'       : part_mass,
          'npart_all'     : self.siminfo["Ngrid"]**3,
          'time'          : theory.z2a(self.ICs["z_ic"]),
          'redshift'      : self.ICs["z_ic"],
          'boxsize'       : self.siminfo["Boxsize"],
          'omegam'        : self.cosmo["Omega_m"],
          'omegal'        : 1.-self.cosmo["Omega_m"],
          'hubble'        : self.cosmo["H0"]*1e-2
        }

        pos = np.column_stack([pos_x, pos_y, pos_z])
        vel = np.column_stack([vel_x, vel_y, vel_z])

        # self.MPI.mpi_print(" -- Processor - " + str(self.rank) + " particle position shape " + str(np.shape(pos)))
        # self.MPI.mpi_print(" -- Processor - " + str(self.rank) + " particle velocity shape " + str(np.shape(vel)))
        # self.MPI.wait()

        fname = self._get_fname_prefix() + 'IC.%i' % self.rank

        self._print_zero()
        self._print_zero(" - Saving ICs in Gadget format to %s[0-%i]"%(fname[:-1], self.MPI.size-1))

        io.save_gadget(fname, header, pos, vel, ic_format=2, single=True, id_offset=part_id_offsets[self.rank])


    def run(self, yaml_fname):
        """Run MIMIC."""
        self.start()
        self.read_paramfile(yaml_fname)
        # Theory
        self.time["Prep_Start"] = time.time()
        self.prep()
        self.compute_eta()
        self.time["Prep_End"] = time.time()

        if self.what2run["WF"]:
            self.time["WF_Start"] = time.time()
            self.get_WF()
            self.time["WF_End"] = time.time()

        if self.what2run["RZA"]:
            self.time["RZA_Start"] = time.time()
            self.get_RZA()
            self.time["RZA_End"] = time.time()

        if self.what2run["IC"]:
            self.time["RR_Start"] = time.time()
            self.get_RR()
            self.time["RR_End"] = time.time()
            if self.what2run["CR"]:
                self.time["CR_Prep_Start"] = time.time()
                self.prep_CR()
                self.time["CR_Prep_End"] = time.time()
                self.time["CR_Start"] = time.time()
                self.get_CR()
                self.time["CR_End"] = time.time()
            self.time["IC_Start"] = time.time()
            self.get_IC()
            self.time["IC_End"] = time.time()
        self.end()


    def _print_time(self, prefix, time_val):
        """Compute print time.

        Parameters
        ----------
        prefix: str
            Prefix to time ouptut.
        time_val : float
            Time.
        """
        if time_val < 0.01:
            self._print_zero(prefix, "%7.4f s" % time_val, " [ %6.2f %% ]" % (100*time_val / (self.time["End"] - self.time["Start"])))
        elif time_val < 1.:
            self._print_zero(prefix, "%7.2f s" % time_val, " [ %6.2f %% ]" % (100*time_val / (self.time["End"] - self.time["Start"])))
        elif time_val < 60:
            self._print_zero(prefix, "%7.2f s" % time_val, " [ %6.2f %% ]" % (100*time_val / (self.time["End"] - self.time["Start"])))
        elif time_val < 60*60:
            self._print_zero(prefix, "%7.2f m" % (time_val/(60.)), " [ %6.2f %% ]" % (100*time_val / (self.time["End"] - self.time["Start"])))
        else:
            self._print_zero(prefix, "%7.2f h" % (time_val/(60.*60.)), " [ %6.2f %% ]" % (100*time_val / (self.time["End"] - self.time["Start"])))


    def end(self):
        """Ends the run."""
        self.MPI.wait()
        self.time["End"] = time.time()

        self._print_zero()
        self._print_zero(" Running Time")
        self._print_zero(" ============")
        self._print_zero()

        Prep_str = " -> Theory Calculations       = "
        WF___str = " -> Wiener Filter             = "
        RZA__str = " -> Reverse Zeldovich         = "
        RR___str = " -> Random Realisation        = "
        PCR__str = " -> CR Preprocessing          = "
        CR___str = " -> Constrained Realisation   = "
        IC___str = " -> Initial Conditions        = "
        TT___str = " -> Total                     = "

        self._print_time(Prep_str, self.time["Prep_End"] - self.time["Prep_Start"])

        if self.what2run["WF"]:
            self._print_time(WF___str, self.time["WF_End"] - self.time["WF_Start"])

        if self.what2run["RZA"]:
            self._print_time(RZA__str, self.time["RZA_End"] - self.time["RZA_Start"])

        if self.what2run["IC"]:
            self._print_time(RR___str, self.time["RR_End"] - self.time["RR_Start"])

            if self.what2run["CR"]:
                self._print_time(PCR__str, self.time["CR_Prep_End"] - self.time["CR_Prep_Start"])
                self._print_time(CR___str, self.time["CR_End"] - self.time["CR_Start"])

            self._print_time(IC___str, self.time["IC_End"] - self.time["IC_Start"])

        self._print_zero()
        self._print_time(TT___str, self.time["End"] - self.time["Start"])

        self._print_zero(mimic_end)
        self.MPI.end()
