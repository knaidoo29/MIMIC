import os
import sys
import time
from typing import Union

import numpy as np

from scipy.interpolate import interp1d
from scipy.sparse.linalg import LinearOperator, gmres, cg, minres, bicgstab

import shift
import fiesta

from ..src import field, io, theory


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
|                          MIMetic Initial Conditions                          |
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

        self.rank = self.MPI.rank

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
            "Sigma8": None,
            "Correct4Sigma8": True,
            "ScaleDepGrowth": None,
            "GrowthFile": None
        }
        self.siminfo = {
            "Boxsize": None,
            "Ngrid": None
        }
        self.constraints = {
            "Fname": None,
            "z_eff": None,
            "Rg": None,
            "Rmax": None,
            "CorrFile": None,
            "CovFile": None,
            "CovOptimise": False,
            "dens_Sigma_NL": 0.,
            "psi_Sigma_NL": 0.,
            "vel_Sigma_NL": 0.,
            "klims": True,
            "gridcorr": True,
            "Ngrid_Large": None
        }
        self.WF = {
            "Field": None,
            "Mode": None,
            "Convert": None,
        }
        self.RZA = {
            "Method": None
        }
        self.ICs = {
            "Seed": None,
            "WNFile": None,
            "z_ic": None,
            "gadget_format": None
        }
        self.outputs = {
            "OutputFolder": None,
            "Prefix": None
        }
        # Need to think about this
        self.what2run = {
            "WF": None,
            "WF_Cons": None,
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
        self.cons_id = None
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
        # subgrid info
        self.sub_x3D = None
        self.sub_y3D = None
        self.sub_z3D = None
        self.sub_x_shape = None
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
        self.xi_box = None
        self.zeta_p_box = None
        self.zeta_u_box = None
        self.psixx_pp_box = None
        self.psixy_pp_box = None
        self.psixx_pu_box = None
        self.psixy_pu_box = None
        self.psixx_uu_box = None
        self.psixy_uu_box = None
        # Covariance related data
        self.cov = None
        self.cov_CR = None
        self.eta = None
        self.eta_CR = None
        # store
        self.dens_WF = None
        self.psi_x_WF = None
        self.psi_y_WF = None
        self.psi_z_WF = None
        self.vel_x_WF = None
        self.vel_y_WF = None
        self.vel_z_WF = None
        self.dens = None
        self.psi_x = None
        self.psi_y = None
        self.psi_z = None
        self.vel_x = None
        self.vel_y = None
        self.vel_z = None
        self._lenpro = 20
        # output
        self.fname_prefix = None

    def start(self):
        """Starts the run and timers."""
        self.time["Start"] = time.time()
        self._print_zero(mimic_beg)

    # Utility functions --------------------------------------------------------

    def _print_zero(self, *value):
        """Print at rank=0."""
        self.MPI.mpi_print_zero(*value)

    def _break4error(self):
        """Forces MIMIC to break if an error is detected."""
        io._break4error(self.ERROR)

    def _check_exist(self, fname):
        """Checks whether a file exists, and breaks if it does not."""
        if io.isfile(fname) is False:
            self.ERROR = True
        io._error_message(self.ERROR, "File %s does not exist" % fname, MPI=self.MPI)
        self._break4error()

    def _get_fname_prefix(self):
        """Returns a filename prefix based on ouput folder and file name prefix
        entered."""
        self.fname_prefix = self.outputs["OutputFolder"]
        if self.rank == 0:
            if io.isfolder(self.fname_prefix) == False:
                io.create_folder(self.fname_prefix)
        if self.fname_prefix[-1] != "/":
            self.fname_prefix += "/"
        self.fname_prefix += self.outputs["Prefix"]
        return self.fname_prefix

    # Parameter file read and management ---------------------------------------

    def _check_param_key(self, params, key):
        """Check param key exists in dictionary, and if key is not None."""
        if key in params:
            if params[key] is not None:
                return True
        else:
            return False

    def _read_params(self, params):
        """Reads parameter file."""

        self._print_zero()
        self._print_zero(" Parameters")
        self._print_zero(" ==========")

        self._print_zero()
        self._print_zero(" MPI:")
        self._print_zero(" -", self.MPI.size, "Processors")

        # Read in Cosmological parameters
        self._print_zero()
        self._print_zero(" Cosmology:")
        self._print_zero()

        if self._check_param_key(params["Cosmology"], "H0"):
            self.cosmo["H0"] = float(params["Cosmology"]["H0"])
            self._print_zero(" - H0                 =", self.cosmo["H0"])
        else:
            self.ERROR = True
            io._error_message(self.ERROR, "H0 must be defined.", MPI=self.MPI)
        self._break4error()

        if self._check_param_key(params["Cosmology"], "Omega_m"):
            self.cosmo["Omega_m"] = float(params["Cosmology"]["Omega_m"])
            self._print_zero(" - Omega_m            =", self.cosmo["Omega_m"])
        else:
            self.ERROR = True
            io._error_message(self.ERROR, "Omega_m must be defined.", MPI=self.MPI)
        self._break4error()

        if self._check_param_key(params["Cosmology"], "PowerSpecFile"):
            self.cosmo["PowerSpecFile"] = str(params["Cosmology"]["PowerSpecFile"])
            self._print_zero(" - PowerSpecFile      =", self.cosmo["PowerSpecFile"])
            self._check_exist(self.cosmo["PowerSpecFile"])
        else:
            self.ERROR = True
            io._error_message(self.ERROR, "PowerSpecFile must be defined.", MPI=self.MPI)
        self._break4error()

        if self._check_param_key(params["Cosmology"], "Sigma8"):
            if params["Cosmology"]["Sigma8"] != 'None':
                self.cosmo["Sigma8"] = params["Cosmology"]["Sigma8"]
                self._print_zero(" - Sigma8             =", self.cosmo["Sigma8"])

        if self._check_param_key(params["Cosmology"], "Correct4Sigma8"):
            if params["Cosmology"]["Correct4Sigma8"] != 'None' :
                self.cosmo["Correct4Sigma8"] = params["Cosmology"]["Correct4Sigma8"]
                self._print_zero(" - Correct4Sigma8     =", self.cosmo["Correct4Sigma8"])

        if self._check_param_key(params["Cosmology"], "ScaleDepGrowth"):
            self.cosmo["ScaleDepGrowth"] = bool(params["Cosmology"]["ScaleDepGrowth"])
            self._print_zero(" - ScaleDepGrowth     =", self.cosmo["ScaleDepGrowth"])
        else:
            self.ERROR = True
            io._error_message(self.ERROR, "ScaleDepGrowth must be defined.", MPI=self.MPI)
        self._break4error()

        if self._check_param_key(params["Cosmology"], "GrowthFile"):
            self.cosmo["GrowthFile"] = str(params["Cosmology"]["GrowthFile"])
            self._print_zero(" - GrowthFile         =", self.cosmo["GrowthFile"])
            self._check_exist(self.cosmo["GrowthFile"])
        else:
            self.ERROR = True
            io._error_message(self.ERROR, "GrowthFile must be defined.", MPI=self.MPI)
        self._break4error()

        # Read in Siminfo
        self._print_zero()
        self._print_zero(" Siminfo:")
        self._print_zero()

        if self._check_param_key(params["Siminfo"], "Boxsize"):
            self.siminfo["Boxsize"] = float(params["Siminfo"]["Boxsize"])
            self._print_zero(" - Boxsize            =", self.siminfo["Boxsize"])
        else:
            self.ERROR = True
            io._error_message(self.ERROR, "Boxsize must be defined.", MPI=self.MPI)
        self._break4error()

        if self._check_param_key(params["Siminfo"], "Ngrid"):
            self.siminfo["Ngrid"] = int(params["Siminfo"]["Ngrid"])
            self._print_zero(" - Ngrid              =", self.siminfo["Ngrid"])
        else:
            self.ERROR = True
            io._error_message(self.ERROR, "Ngrid must be defined.", MPI=self.MPI)
        self._break4error()

        # Read in Constraints
        if self._check_param_key(params, "Constraints"):

            self._print_zero()
            self._print_zero(" Constraints:")
            self._print_zero()

            if self._check_param_key(params["Constraints"], "Fname"):
                self.constraints["Fname"] = str(params["Constraints"]["Fname"])
                self._print_zero(" - Fname              =", self.constraints["Fname"])
                self._check_exist(self.constraints["Fname"])
            else:
                self.ERROR = True
                io._error_message(self.ERROR, "Constraint Fname must be defined.", MPI=self.MPI)
            self._break4error()

            if self._check_param_key(params["Constraints"], "z_eff"):
                self.constraints["z_eff"] = float(params["Constraints"]["z_eff"])
                self._print_zero(" - z_eff              =", self.constraints["z_eff"])
            else:
                self.ERROR = True
                io._error_message(self.ERROR, "Constraint z_eff must be defined.", MPI=self.MPI)
            self._break4error()

            if self._check_param_key(params["Constraints"], "Rg"):
                self.constraints["Rg"] = float(params["Constraints"]["Rg"])
                self._print_zero(" - Rg                 =", self.constraints["Rg"])
            else:
                self.ERROR = True
                io._error_message(self.ERROR, "Constraint Rg must be defined.", MPI=self.MPI)
            self._break4error()

            if self._check_param_key(params["Constraints"], "Rmax"):
                if params["Constraints"]["Rmax"] != 'None':
                    self.constraints["Rmax"] = params["Constraints"]["Rmax"]
                    self._print_zero(" - Rmax               =", self.constraints["Rmax"])

            if self._check_param_key(params["Constraints"], "CorrFile"):
                if params["Constraints"]["CorrFile"] != 'None':
                    self.constraints["CorrFile"] = params["Constraints"]["CorrFile"]
                    self._check_exist(self.constraints["CorrFile"])
                    self._print_zero(" - CorrFile           =", self.constraints["CorrFile"])

            if self._check_param_key(params["Constraints"], "CovFile"):
                if str(params["Constraints"]["CovFile"]) != "None":
                    self.constraints["CovFile"] = params["Constraints"]["CovFile"]
                    self._check_exist(self.constraints["CovFile"])
                    self._print_zero(" - CovFile            =", self.constraints["CovFile"])

            if self._check_param_key(params["Constraints"], "CovOptimise"):
                if params["Constraints"]["CovOptimise"] != "None":
                    self.constraints["CovOptimise"] = bool(params["Constraints"]["CovOptimise"])
                    self._print_zero(" - CovOptimise        =", io.bool2yesno(self.constraints["CovOptimise"]))

            if self._check_param_key(params["Constraints"], "dens_Sigma_NL"):
                if params["Constraints"]["dens_Sigma_NL"] != 'None':
                    self.constraints["dens_Sigma_NL"] = float(params["Constraints"]["dens_Sigma_NL"])
                    self._print_zero(" - dens_Sigma_NL      =", self.constraints["dens_Sigma_NL"])

            if self._check_param_key(params["Constraints"], "psi_Sigma_NL"):
                if params["Constraints"]["psi_Sigma_NL"] != "None":
                    self.constraints["psi_Sigma_NL"] = float(params["Constraints"]["psi_Sigma_NL"])
                    self._print_zero(" - psi_Sigma_NL       =", self.constraints["psi_Sigma_NL"])

            if self._check_param_key(params["Constraints"], "vel_Sigma_NL"):
                if params["Constraints"]["vel_Sigma_NL"] != "None":
                    self.constraints["vel_Sigma_NL"] = float(params["Constraints"]["vel_Sigma_NL"])
                    self._print_zero(" - vel_Sigma_NL       =", self.constraints["vel_Sigma_NL"])

            if self._check_param_key(params["Constraints"], "klims"):
                if params["Constraints"]["klims"] != "None":
                    self.constraints["klims"] = bool(params["Constraints"]["klims"])
                    self._print_zero(" - klims              =", io.bool2yesno(self.constraints["klims"]))

            if self._check_param_key(params["Constraints"], "gridcorr"):
                if params["Constraints"]["gridcorr"] != "None":
                    self.constraints["gridcorr"] = bool(params["Constraints"]["gridcorr"])
                    self._print_zero(" - gridcorr           =", io.bool2yesno(self.constraints["gridcorr"]))

            if self._check_param_key(params["Constraints"], "Ngrid_Large"):
                if params["Constraints"]["Ngrid_Large"] != "None":
                    self.constraints["Ngrid_Large"] = int(params["Constraints"]["Ngrid_Large"])
                    self._print_zero(" - Ngrid_Large        =", self.constraints["Ngrid_Large"])

        if self._check_param_key(params, "WF"):

            self._print_zero()
            self._print_zero(" WF:")

            self.WF["Field"] = params["WF"]["Field"]

            check = io.inlist(self.WF["Field"], ["dens", "psi_x", "psi_y", "psi_z", "psi_r", "vel_x", "vel_y", "vel_z", "vel_r"])

            self.ERROR = io._error_if_false(check)
            io._error_message(self.ERROR, "Field string is unsupported, current %s but must be either 'dens', 'psi_x', 'psi_y', 'psi_z', 'vel_x', 'vel_y' and 'vel_z'.")

            self._print_zero(" - Field              =", self.WF["Field"])

            if self._check_param_key(params["WF"], "Mode"):
                self.WF["Mode"] = params["WF"]["Mode"]
                self._print_zero(" - Mode               =", self.WF["Mode"])
            else:
                self.ERROR = True
            io._error_message(self.ERROR, "WF Mode must be defined.", MPI=self.MPI)
            self._break4error()

            if self.WF["Mode"] == "Full":

                self.what2run["WF"] = True

                if self._check_param_key(params["WF"], "Convert"):
                    if params["WF"]["Convert"] != "None":
                        if self.WF["Field"] == "dens":
                            if io.inlist(params["WF"]["Convert"], ["psi", "vel"]):
                                self.WF["Convert"] = params["WF"]["Convert"]
                                self._print_zero(" - Convert            =", self.WF["Convert"])
                            else:
                                self.ERROR = True
                                io._error_message(self.ERROR, "WF convert %s must either be 'psi' or 'vel'." % self.WF["Convert"], MPI=self.MPI)
                        else:
                            self.ERROR = True
                            io._error_message(self.ERROR, "WF convert only supported for Field='dens'.", MPI=self.MPI)
                    self._break4error()

        if self._check_param_key(params, "RZA"):

            self._print_zero()
            self._print_zero(" RZA:")

            if self._check_param_key(params["RZA"], "Method"):
                self.RZA["Method"] = int(params["RZA"]["Method"])
                self._print_zero(" - Method             =", self.RZA["Method"])
            else:
                self.ERROR = True
            io._error_message(self.ERROR, "RZA method must be defined.", MPI=self.MPI)
            self._break4error()

            if self.WF["Field"] != "dens":
                self.ERROR = True

            io._error_message(self.ERROR, "Field must be density ('dens') if RZA is required", MPI=self.MPI)
            self._break4error()

            self.what2run["WF"] = True
            self.what2run["RZA"] = True

        # ICs
        if self._check_param_key(params, "ICs"):

            self._print_zero()
            self._print_zero(" ICs:")

            if self._check_param_key(params["ICs"], "Seed"):
                self.ICs["Seed"] = int(params["ICs"]["Seed"])
                self._print_zero()
                self._print_zero(" - Seed               =", self.ICs["Seed"])
            elif self._check_param_key(params["ICs"], "WNFile"):
                if params["ICs"]["WNFile"] != "None":
                    self.ICs["WNFile"] = str(params["ICs"]["WNFile"])
                    self._check_exist(self.ICs["WNFile"])
                    self._break4error()
                    self._print_zero()
                    self._print_zero(" - WNFile             =", self.ICs["WNFile"])
                else:
                    self.ERROR = True
            else:
                self.ERROR = True

            io._error_message(self.ERROR, "Must specify either a Seed or WNFile.", MPI=self.MPI)
            self._break4error()

            if self._check_param_key(params["ICs"], "z_ic"):
                self.ICs["z_ic"] = float(params["ICs"]["z_ic"])
                self._print_zero(" - z_ic               =", self.ICs["z_ic"])
            else:
                self.ERROR = True
                io._error_message(self.ERROR, "IC z_ic must be defined.", MPI=self.MPI)
            self._break4error()

            if self._check_param_key(params["ICs"], "gadget_format"):
                self.ICs["gadget_format"] = int(params["ICs"]["gadget_format"])
                self._print_zero(" - gadget_format      =", self.ICs["gadget_format"])
            else:
                self.ERROR = True
                io._error_message(self.ERROR, "IC gadget_format must be defined.", MPI=self.MPI)
            self._break4error()

            if self._check_param_key(params["ICs"], "CR"):
                self.what2run["CR"] = bool(params["ICs"]["CR"])
            else:
                self.what2run["CR"] = False

            self.what2run["IC"] = True

            self._print_zero(" - CR                 =", io.bool2yesno(self.what2run["CR"]))
            self._print_zero(" - IC                 =", io.bool2yesno(self.what2run["IC"]))

        else:
            self.what2run["IC"] = False

        # Outputs
        self._print_zero()
        self._print_zero(" Outputs:")
        self._print_zero()

        if self._check_param_key(params["Outputs"], "OutputFolder"):
            self.outputs["OutputFolder"] = str(params["Outputs"]["OutputFolder"])
            self._print_zero(" - OutputFolder       =", self.outputs["OutputFolder"])
        else:
            self.ERROR = True
        io._error_message(self.ERROR, "OutputFolder must be defined.", MPI=self.MPI)
        self._break4error()

        if self._check_param_key(params["Outputs"], "Prefix"):
            self.outputs["Prefix"] = str(params["Outputs"]["Prefix"])
            self._print_zero(" - Prefix             =", self.outputs["Prefix"])
        else:
            self.ERROR = True
        io._error_message(self.ERROR, "Output prefix must be defined.", MPI=self.MPI)
        self._break4error()


    def read_paramfile(self, yaml_fname):
        """Reads parameter file."""
        self.params, self.ERROR = io.read_paramfile(yaml_fname, MPI=self.MPI)
        self._break4error()
        self._read_params(self.params)
        self._break4error()

    # MPI utility functions -------------------------------------------------------

    def MPI_create_split_ndarray(self, MPI, arrays_nd, whichaxis):
        """Split a list of arrays based on the data partitioning scheme."""
        split_arrays = []
        for i in range(0, len(arrays_nd)):
            _array = arrays_nd[i]
            if not whichaxis[i]:
                _array = MPI.split_array(_array)
                split_arrays.append(_array)
            else:
                split_arrays.append(_array)
        return split_arrays


    def MPI_create_split_ndgrid(self, MPI, arrays_nd, whichaxis):
        """Create a partitioned gridded data set."""
        split_arrays = self.MPI_create_split_ndarray(MPI, arrays_nd, whichaxis)
        split_grid = np.meshgrid(*split_arrays, indexing='ij')
        return split_grid

    # Theory Calculations ------------------------------------------------------

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
        """Prepares MPI grid partitioning class."""
        self.SBX = fiesta.coords.MPI_SortByX(self.MPI)
        self.SBX.settings(self.siminfo["Boxsize"], self.siminfo["Ngrid"])
        self.SBX.limits4grid()

    def _prep_theory(self):
        """Loads theory related functions, i.e. power spectra, expansion rate
        and growth functions."""
        self._print_zero()
        self._print_zero(" Theory")
        self._print_zero(" ======")
        self._print_zero()

        self._print_zero(" - Load PowerSpecFile :", self.cosmo["PowerSpecFile"])
        data = np.load(self.cosmo["PowerSpecFile"])
        self.theory_kh, self.theory_pk = data['kh'], data['pk']

        if self.cosmo["Correct4Sigma8"]:
            self._print_zero(" - Correct P(k) for Sigma8")
            sigma8_frompk = theory.get_sigma_8(self.theory_kh, self.theory_pk)
            self._print_zero(" -- Sigma8 from Pk = %0.4f" % sigma8_frompk)
            kf = shift.cart.get_kf(self.siminfo["Boxsize"])
            kn = shift.cart.get_kn(self.siminfo["Boxsize"], self.siminfo["Ngrid"])
            cond = np.where((self.theory_kh >= kf) & (self.theory_kh <= kn))[0]
            sigma8_frompklim = theory.get_sigma_8(self.theory_kh[cond], self.theory_pk[cond])
            self._print_zero(" -- Sigma8 from limited Pk = %0.4f" % sigma8_frompklim)
            self._print_zero(" -- Sigma8 correction factor = %0.4f" % (sigma8_frompk/sigma8_frompklim)**2.)
            self.theory_pk *= (sigma8_frompk/sigma8_frompklim)**2.

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

    # Constraints management ---------------------------------------------------

    def _check_constraints(self):
        """Check constraints are within the box."""
        self._print_zero()
        self._print_zero(" - Prepare Constraints")
        # Force type conversion to prevent odd 'float' has not 'sqrt' attribute error.
        # Not quite sure why this is necessary, but this type specification seems to
        # fix the issue.
        self.cons_x = self.cons_x.astype('float')
        self.cons_y = self.cons_y.astype('float')
        self.cons_z = self.cons_z.astype('float')
        self.cons_ex = self.cons_ex.astype('float')
        self.cons_ey = self.cons_ey.astype('float')
        self.cons_ez = self.cons_ez.astype('float')
        self.cons_id = self.cons_id.astype('int')
        self.cons_c = self.cons_c.astype('float')
        self.cons_c_err = self.cons_c_err.astype('float')
        self.cons_c_type = self.cons_c_type.astype('int')
        if self.cons_c_RR is not None:
            self.cons_c_RR = self.cons_c_RR.astype('float')
        # Normalise direction
        self._print_zero(" -- Normalize velocity unit vector")
        norm = (self.cons_ex**2. + self.cons_ey**2. + self.cons_ez**2.)**0.5
        self.cons_ex /= norm
        self.cons_ey /= norm
        self.cons_ez /= norm
        # Keep only positions inside the box, r <= halfboxsize
        self._print_zero(" -- Remove constrained points outside of the simulation box")
        if self.constraints["Rmax"] is None:
            cond = np.where((self.cons_x >= 0.) & (self.cons_x <= self.siminfo["Boxsize"]) &
                            (self.cons_y >= 0.) & (self.cons_y <= self.siminfo["Boxsize"]) &
                            (self.cons_z >= 0.) & (self.cons_z <= self.siminfo["Boxsize"]))[0]
        else:
            cons_r2 = (self.cons_x - self.halfsize)**2.
            cons_r2 += (self.cons_y - self.halfsize)**2.
            cons_r2 += (self.cons_z - self.halfsize)**2.
            cons_r = np.sqrt(cons_r2)
            cond = np.where((self.cons_x >= 0.) & (self.cons_x <= self.siminfo["Boxsize"]) &
                            (self.cons_y >= 0.) & (self.cons_y <= self.siminfo["Boxsize"]) &
                            (self.cons_z >= 0.) & (self.cons_z <= self.siminfo["Boxsize"]) &
                            (cons_r <= self.constraints["Rmax"]))[0]
        self._print_zero(" -- Retained %i constrained points from %i" % (len(cond), len(self.cons_x)))
        self.cons_id = self.cons_id[cond]
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
        """Preparing constraints, i.e. loading and moving to the center of the box."""
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
        self.cons_id = np.arange(len(self.cons_x))
        self._check_constraints()


    # Correlation functions ----------------------------------------------------

    def _save_correlators(self):
        """Save correlation functions."""
        if self.rank == 0:
            fname_prefix = self._get_fname_prefix()
            fname = fname_prefix + "analytic_correlator.npz"
            self._print_zero(" - Save analytic correlation function as :", fname)
            io.save_analytic_correlators(fname, self.corr_redshift, self.corr_r, self.corr_xi,
                self.corr_zeta_p, self.corr_zeta_u, self.corr_psiR_pp, self.corr_psiT_pp,
                self.corr_psiR_pu, self.corr_psiT_pu, self.corr_psiR_uu, self.corr_psiT_uu,
                filetype='npz')
            fname = fname_prefix + "grid_correlator.npz"
            self._print_zero(" - Save grid correlation function as :", fname)
            io.save_grid_correlators(fname, self.corr_redshift, self.xi_box,
                self.zeta_p_box, self.zeta_u_box, self.psixx_pp_box, self.psixy_pp_box,
                self.psixx_pu_box, self.psixy_pu_box, self.psixx_uu_box, self.psixy_uu_box, self.stretch_grid,
                filetype='npz')


    def _calc_analytic_correlators(self):
        """Calculate analytic correlation functions."""

        self._print_zero(" - Computing analytic correlators in parallel")

        if self.constraints["klims"]:

            self.sim_kmin = None
            self.sim_kmax = None

            kf = shift.cart.get_kf(self.siminfo["Boxsize"])
            kn = shift.cart.get_kn(self.siminfo["Boxsize"], self.siminfo["Ngrid"])

            smallfilter = field.get_lowres_filter(self.theory_kh, kn, k0=None, T=0.1)
            largefilter = field.get_highres_filter(self.theory_kh, kf, k0=None, T=0.1)

        else:
            self.sim_kmin = None
            self.sim_kmax = None
            smallfilter = 1.
            largefilter = 1.

        self.corr_r = np.logspace(-2, np.log10(np.sqrt(3.)*self.siminfo["Boxsize"]), 1000)

        Dz2 = self._get_growth_D(self.corr_redshift, kmag=self.theory_kh)**2.
        fz0 = self._get_growth_f(self.corr_redshift, kmag=self.theory_kh)

        _corr_r = self.MPI.split_array(self.corr_r)
        _Rg = self.constraints["Rg"]

        self._print_zero(" -- Computing xi(r)")

        _xi = theory.pk2xi(
            _corr_r, self.theory_kh, Dz2*self.theory_pk*smallfilter*largefilter, 
            kmin=self.sim_kmin, kmax=self.sim_kmax, kfactor=100, kbinsmin=int(1e4), 
            kbinsmax=int(1e6), Rg=_Rg
        )

        self._print_zero(" -- Computing zeta^p(r)")

        _zeta_p = theory.pk2zeta(
            _corr_r, self.theory_kh, Dz2*self.theory_pk*smallfilter*largefilter, 
            fk=None, kmin=self.sim_kmin, kmax=self.sim_kmax, kfactor=100, kbinsmin=int(1e4),
            kbinsmax=int(1e6), Rg=_Rg
        )

        self._print_zero(" -- Computing zeta^u(r)")

        if self.cosmo["ScaleDepGrowth"]:
            _zeta_u = theory.pk2zeta(
                _corr_r, self.theory_kh, Dz2*self.theory_pk*smallfilter*largefilter, 
                fk=fz0, kmin=self.sim_kmin, kmax=self.sim_kmax, kfactor=100, kbinsmin=int(1e4), 
                kbinsmax=int(1e6), Rg=_Rg
            )
        else:
            _zeta_u = fz0*np.copy(_zeta_p)

        self._print_zero(" -- Computing psiR^pp(r) and psiT^pp(r)")

        _psiR_pp = theory.pk2psiR(
            _corr_r, self.theory_kh, Dz2*self.theory_pk*smallfilter*largefilter, 
            fk=None, kmin=self.sim_kmin, kmax=self.sim_kmax,
            kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6), Rg=_Rg
        )
        _psiT_pp = theory.pk2psiT(
            _corr_r, self.theory_kh, Dz2*self.theory_pk*smallfilter*largefilter, 
            fk=None, kmin=self.sim_kmin, kmax=self.sim_kmax,
            kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6), Rg=_Rg
        )

        self._print_zero(" -- Computing psiR^pu(r) and psiT^pu(r)")

        if self.cosmo["ScaleDepGrowth"]:
            _psiR_pu = theory.pk2psiR(
                _corr_r, self.theory_kh, Dz2*self.theory_pk*smallfilter*largefilter, 
                fk=np.sqrt(fz0), kmin=self.sim_kmin, kmax=self.sim_kmax,
                kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6), Rg=_Rg
            )
            _psiT_pu = theory.pk2psiT(
                _corr_r, self.theory_kh, Dz2*self.theory_pk*smallfilter*largefilter, 
                fk=np.sqrt(fz0), kmin=self.sim_kmin, kmax=self.sim_kmax,
                kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6), Rg=_Rg
            )
        else:
            _psiR_pu = fz0*np.copy(_psiR_pp)
            _psiT_pu = fz0*np.copy(_psiT_pp)

        self._print_zero(" -- Computing psiR^uu(r) and psiT^uu(r)")

        if self.cosmo["ScaleDepGrowth"]:
            _psiR_uu = theory.pk2psiR(
                _corr_r, self.theory_kh, Dz2*self.theory_pk*smallfilter*largefilter, 
                fk=fz0, kmin=self.sim_kmin, kmax=self.sim_kmax,
                kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6), Rg=_Rg
                )
            _psiT_uu = theory.pk2psiT(
                _corr_r, self.theory_kh, Dz2*self.theory_pk*smallfilter*largefilter, 
                fk=fz0, kmin=self.sim_kmin, kmax=self.sim_kmax,
                kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6), Rg=_Rg
            )
        else:
            _psiR_uu = (fz0**2)*np.copy(_psiR_pp)
            _psiT_uu = (fz0**2)*np.copy(_psiT_pp)

        _psiR_uu = theory.pk2psiR(
            _corr_r, self.theory_kh, Dz2*self.theory_pk*smallfilter*largefilter, 
            fk=fz0, kmin=self.sim_kmin, kmax=self.sim_kmax,
            kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6), Rg=_Rg
        )
        _psiT_uu = theory.pk2psiT(
            _corr_r, self.theory_kh, Dz2*self.theory_pk*smallfilter*largefilter, 
            fk=fz0, kmin=self.sim_kmin, kmax=self.sim_kmax,
            kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6), Rg=_Rg
        )

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
    

    def _calc_grid_correlators(self):
        """
        Calculate grid correlation functions.
        """

        self._print_zero(" - Compute grid correlators at each rank")
        self._print_zero(" -- Compute p(k) interpolator")
        
        kn = shift.cart.get_kn(self.siminfo["Boxsize"], self.siminfo["Ngrid"])

        smallfilter = 1.#field.get_lowres_filter(self.theory_kh, kn, k0=None, T=0.1)

        Dz2 = self._get_growth_D(self.corr_redshift, kmag=self.theory_kh)**2.
        fz0 = self._get_growth_f(self.corr_redshift, kmag=self.theory_kh)

        pk_interp = interp1d(
            self.theory_kh, Dz2*self.theory_pk*smallfilter, kind='cubic', 
            bounds_error=False, fill_value=0.
        )

        if io.isscalar(fz0) == False:
            fz_interp = interp1d(
                self.theory_kh, fz0, kind='cubic', bounds_error=False, fill_value=0.
            )
            fk3d = fz_interp(kmag)
        else:
            fk3d = fz0

        self._print_zero(" -- Compute p(k) grid")

        self.get_kgrid3D()
        kmag = np.sqrt(self.kx3D**2. + self.ky3D**2. + self.kz3D**2.)

        pk3d = pk_interp(kmag)*(np.ones_like(kmag) + 0j*np.ones_like(kmag))
        # to get the correct normalisation (1/(2*pi)^3) for the FFT convention used in shift.cart.ifft3D
        pk3d /= (np.sqrt(2*np.pi))**3
        
        # dx = self.siminfo["Boxsize"] / self.siminfo["Ngrid"]

        # pk3d = pk_interp(kmag).astype(np.complex128)
        # pk3d *= 1.0 / dx**3

        cond = np.where(kmag != 0.)

        self._print_zero(" -- Compute xi grid")

        self.xi_box = shift.cart.mpi_ifft3D(pk3d, self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)

        self._print_zero(" -- Compute zeta gridr")

        zetak = np.copy(pk3d)
        zetak[cond] *= -1j*self.kx3D[cond]/(kmag[cond]**2.)
        self.zeta_p_box = shift.cart.mpi_ifft3D(zetak, self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)

        zetak = np.copy(fk3d*pk3d)
        zetak[cond] *= -1j*self.kx3D[cond]/(kmag[cond]**2.)
        self.zeta_u_box = shift.cart.mpi_ifft3D(zetak, self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)

        self._print_zero(" -- Compute Psi_xx grid")

        psixxk = np.copy(pk3d)
        psixxk[cond] *= -1j*(self.kx3D[cond]**2.)/(kmag[cond]**4.)
        self.psixx_pp_box = shift.cart.mpi_ifft3D(psixxk, self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)

        psixxk = np.copy(pk3d*fk3d)
        psixxk[cond] *= -1j*(self.kx3D[cond]**2.)/(kmag[cond]**4.)
        self.psixx_pu_box = shift.cart.mpi_ifft3D(psixxk, self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)

        psixxk = np.copy(pk3d*fk3d*fk3d)
        psixxk[cond] *= -1j*(self.kx3D[cond]**2.)/(kmag[cond]**4.)
        self.psixx_uu_box = shift.cart.mpi_ifft3D(psixxk, self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)

        self._print_zero(" -- Compute Psi_xy grid")

        psixyk = np.copy(pk3d)
        psixyk[cond] *= -1j*(self.kx3D[cond]*self.ky3D[cond])/(kmag[cond]**4.)
        self.psixy_pp_box = shift.cart.mpi_ifft3D(psixyk, self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)

        psixyk = np.copy(pk3d*fk3d)
        psixyk[cond] *= -1j*(self.kx3D[cond]*self.ky3D[cond])/(kmag[cond]**4.)
        self.psixy_pu_box = shift.cart.mpi_ifft3D(psixyk, self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)

        psixyk = np.copy(pk3d*fk3d*fk3d)
        psixyk[cond] *= -1j*(self.kx3D[cond]*self.ky3D[cond])/(kmag[cond]**4.)
        self.psixy_uu_box = shift.cart.mpi_ifft3D(psixyk, self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)

        if self.constraints["Ngrid_Large"] is not None:

            self._print_zero(" -- Interpolate grid onto stretched sin grid")

            xcos, ycos, zcos = shift.cart.grid3D(1., self.constraints["Ngrid_Large"])

            _, xcosgrid = shift.cart.grid1D(1., self.constraints["Ngrid_Large"])

            xstretchgrid = theory.stretch_sin_backward(xcosgrid, self.siminfo["Boxsize"])

            xstretch = theory.stretch_sin_backward(xcos, self.siminfo["Boxsize"])
            ystretch = theory.stretch_sin_backward(ycos, self.siminfo["Boxsize"])
            zstretch = theory.stretch_sin_backward(zcos, self.siminfo["Boxsize"])
            
            xedges, _ = shift.cart.mpi_grid1D(self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)

            dx = xedges[1]-xedges[0]
            xmin = xedges[0]
            xmax = xedges[-1]

            cond = np.where((xstretchgrid >= xmin) & (xstretchgrid <= xmax))[0]

            xstretch = xstretch[cond]
            ystretch = ystretch[cond]
            zstretch = zstretch[cond]

            xsshape = np.shape(xstretch)
            
            xstretch = xstretch.flatten()
            ystretch = ystretch.flatten()
            zstretch = zstretch.flatten()

            xmin -= dx
            xmax += dx

            self.xi_box_up = self.MPI.send_up(self.xi_box[-1]) 
            self.xi_box_down = self.MPI.send_down(self.xi_box[0])
            self.xi_box = np.concatenate([np.array([self.xi_box_up]), self.xi_box, np.array([self.xi_box_down])], axis=0) 

            self.zeta_u_box_up = self.MPI.send_up(self.zeta_u_box[-1]) 
            self.zeta_u_box_down = self.MPI.send_down(self.zeta_u_box[0])
            self.zeta_u_box = np.concatenate([np.array([self.zeta_u_box_up]), self.zeta_u_box, np.array([self.zeta_u_box_down])], axis=0) 

            self.zeta_p_box_up = self.MPI.send_up(self.zeta_p_box[-1]) 
            self.zeta_p_box_down = self.MPI.send_down(self.zeta_p_box[0])
            self.zeta_p_box = np.concatenate([np.array([self.zeta_p_box_up]), self.zeta_p_box, np.array([self.zeta_p_box_down])], axis=0) 

            self.psixx_pp_box_up = self.MPI.send_up(self.psixx_pp_box[-1])
            self.psixx_pp_box_down = self.MPI.send_down(self.psixx_pp_box[0])
            self.psixx_pp_box = np.concatenate([np.array([self.psixx_pp_box_up]), self.psixx_pp_box, np.array([self.psixx_pp_box_down])], axis=0)

            self.psixy_pp_box_up = self.MPI.send_up(self.psixy_pp_box[-1])
            self.psixy_pp_box_down = self.MPI.send_down(self.psixy_pp_box[0])
            self.psixy_pp_box = np.concatenate([np.array([self.psixy_pp_box_up]), self.psixy_pp_box, np.array([self.psixy_pp_box_down])], axis=0)

            self.psixx_pu_box_up = self.MPI.send_up(self.psixx_pu_box[-1])
            self.psixx_pu_box_down = self.MPI.send_down(self.psixx_pu_box[0])
            self.psixx_pu_box = np.concatenate([np.array([self.psixx_pu_box_up]), self.psixx_pu_box, np.array([self.psixx_pu_box_down])], axis=0)

            self.psixy_pu_box_up = self.MPI.send_up(self.psixy_pu_box[-1])
            self.psixy_pu_box_down = self.MPI.send_down(self.psixy_pu_box[0])
            self.psixy_pu_box = np.concatenate([np.array([self.psixy_pu_box_up]), self.psixy_pu_box, np.array([self.psixy_pu_box_down])], axis=0)

            self.psixx_uu_box_up = self.MPI.send_up(self.psixx_uu_box[-1])
            self.psixx_uu_box_down = self.MPI.send_down(self.psixx_uu_box[0])
            self.psixx_uu_box = np.concatenate([np.array([self.psixx_uu_box_up]), self.psixx_uu_box, np.array([self.psixx_uu_box_down])], axis=0)

            self.psixy_uu_box_up = self.MPI.send_up(self.psixy_uu_box[-1])
            self.psixy_uu_box_down = self.MPI.send_down(self.psixy_uu_box[0])
            self.psixy_uu_box = np.concatenate([np.array([self.psixy_uu_box_up]), self.psixy_uu_box, np.array([self.psixy_uu_box_down])], axis=0)

            self.xi_box = fiesta.interp.trilinear(
                self.xi_box, [xmax-xmin, self.siminfo["Boxsize"], self.siminfo["Boxsize"]],
                xstretch-xmin, ystretch, zstretch, fill_value=np.nan, periodic=[False, True, True]
            )

            self.zeta_p_box = fiesta.interp.trilinear(
                self.zeta_p_box, [xmax-xmin, self.siminfo["Boxsize"], self.siminfo["Boxsize"]],
                xstretch-xmin, ystretch, zstretch, fill_value=np.nan, periodic=[False, True, True]
            )

            self.zeta_u_box = fiesta.interp.trilinear(
                self.zeta_u_box, [xmax-xmin, self.siminfo["Boxsize"], self.siminfo["Boxsize"]],
                xstretch-xmin, ystretch, zstretch, fill_value=np.nan, periodic=[False, True, True]
            )

            self.psixx_pp_box = fiesta.interp.trilinear(
                self.psixx_pp_box, [xmax-xmin, self.siminfo["Boxsize"], self.siminfo["Boxsize"]],
                xstretch-xmin, ystretch, zstretch, fill_value=np.nan, periodic=[False, True, True]
            )

            self.psixy_pp_box = fiesta.interp.trilinear(
                self.psixy_pp_box, [xmax-xmin, self.siminfo["Boxsize"], self.siminfo["Boxsize"]],
                xstretch-xmin, ystretch, zstretch, fill_value=np.nan, periodic=[False, True, True]
            )
            self.psixx_pu_box = fiesta.interp.trilinear(
                self.psixx_pu_box, [xmax-xmin, self.siminfo["Boxsize"], self.siminfo["Boxsize"]],
                xstretch-xmin, ystretch, zstretch, 
                fill_value=np.nan, periodic=[False, True, True]
            )

            self.psixy_pu_box = fiesta.interp.trilinear(
                self.psixy_pu_box, [xmax-xmin, self.siminfo["Boxsize"], self.siminfo["Boxsize"]],
                xstretch-xmin, ystretch, zstretch, 
                fill_value=np.nan, periodic=[False, True, True]
            )

            self.psixx_uu_box = fiesta.interp.trilinear(
                self.psixx_uu_box, [xmax-xmin, self.siminfo["Boxsize"], self.siminfo["Boxsize"]],
                xstretch-xmin, ystretch, zstretch, 
                fill_value=np.nan, periodic=[False, True, True]
            )

            self.psixy_uu_box = fiesta.interp.trilinear(
                self.psixy_uu_box, [xmax-xmin, self.siminfo["Boxsize"], self.siminfo["Boxsize"]],
                xstretch-xmin, ystretch, zstretch, 
                fill_value=np.nan, periodic=[False, True, True]
            )

            self.xi_box = self.xi_box.reshape(xsshape)
            self.zeta_p_box = self.zeta_p_box.reshape(xsshape)
            self.zeta_u_box = self.zeta_u_box.reshape(xsshape)
            self.psixx_pp_box = self.psixx_pp_box.reshape(xsshape)
            self.psixy_pp_box = self.psixy_pp_box.reshape(xsshape)
            self.psixx_pu_box = self.psixx_pu_box.reshape(xsshape)
            self.psixy_pu_box = self.psixy_pu_box.reshape(xsshape)
            self.psixx_uu_box = self.psixx_uu_box.reshape(xsshape)
            self.psixy_uu_box = self.psixy_uu_box.reshape(xsshape)

            self.stretch_grid = True
        else:
            self.stretch_grid = False
        
        self._print_zero(" -- Collect and broadcast full grid correlator boxes")

        self.xi_box = self.MPI.collect(self.xi_box)
        self.zeta_p_box = self.MPI.collect(self.zeta_p_box)
        self.zeta_u_box = self.MPI.collect(self.zeta_u_box)

        self.psixx_pp_box = self.MPI.collect(self.psixx_pp_box)
        self.psixy_pp_box = self.MPI.collect(self.psixy_pp_box)

        self.psixx_pu_box = self.MPI.collect(self.psixx_pu_box)
        self.psixy_pu_box = self.MPI.collect(self.psixy_pu_box)

        self.psixx_uu_box = self.MPI.collect(self.psixx_uu_box)
        self.psixy_uu_box = self.MPI.collect(self.psixy_uu_box)

        self.xi_box = self.MPI.broadcast(self.xi_box)
        self.zeta_p_box = self.MPI.broadcast(self.zeta_p_box)
        self.zeta_u_box = self.MPI.broadcast(self.zeta_u_box)

        self.psixx_pp_box = self.MPI.broadcast(self.psixx_pp_box)
        self.psixy_pp_box = self.MPI.broadcast(self.psixy_pp_box)

        self.psixx_pu_box = self.MPI.broadcast(self.psixx_pu_box)
        self.psixy_pu_box = self.MPI.broadcast(self.psixy_pu_box)

        self.psixx_uu_box = self.MPI.broadcast(self.psixx_uu_box)
        self.psixy_uu_box = self.MPI.broadcast(self.psixy_uu_box)


    def _prep_correlators(self, redshift):
        """Constructing correlation interpolation functions."""
        self._print_zero()
        self._print_zero(" Correlators")
        self._print_zero(" ===========")
        self._print_zero()

        self.corr_redshift = redshift

        self._calc_analytic_correlators()
        self._print_zero()
        self._calc_grid_correlators()
        self._print_zero()
        self._save_correlators()


    # This might need some rethinking, naming wise rather than pipeline.
    def prep(self):
        """Runs all the grid, theory and constraint preparation functions."""
        self._prep_grid()
        self._prep_theory()
        self._prep_constraints()
        self._prep_correlators(self.constraints["z_eff"])


    # Distributed matrix solve helper functions ---------------------------

    def _add_diag_to_local_rows(self, A_local, row_ind, diag):
        """Add a full diagonal vector to a row-distributed matrix.

        A_local contains only this rank's rows, but all columns.
        row_ind gives the global row index for each local row.
        diag is the full global diagonal vector.
        """
        row_ind = np.asarray(row_ind, dtype=np.int64)
        diag = np.asarray(diag, dtype=np.float64)

        for iloc, iglob in enumerate(row_ind):
            A_local[iloc, iglob] += diag[iglob]


    def _add_scalar_to_local_diagonal(self, A_local, row_ind, lam):
        """Add lam * I to a row-distributed matrix."""
        row_ind = np.asarray(row_ind, dtype=np.int64)

        for iloc, iglob in enumerate(row_ind):
            A_local[iloc, iglob] += lam


    def _matvec_rowdist(self, A_local, x):
        """Distributed matrix-vector product y = A x.

        Parameters
        ----------
        A_local : ndarray
            Local row slab of the matrix, with shape (n_local_rows, n_total).
        x : ndarray
            Full input vector, replicated on all ranks.

        Returns
        -------
        y : ndarray
            Full output vector, replicated on all ranks.

        Notes
        -----
        The matrix remains distributed. Only vectors are replicated.
        """
        x = np.asarray(x, dtype=np.float64)

        # Each rank computes its owned rows.
        y_local = A_local.dot(x)

        y = self.MPI.collect(y_local)
        y = self.MPI.broadcast(y)

        return np.asarray(y, dtype=np.float64)


    def _solve_eta_rowdist_gmres(
        self,
        A_local,
        b,
        tol=1e-8,
        atol=0.0,
        restart=50,
        maxiter=500,
    ):
        """Solve A eta = b using GMRES with row-distributed A.

        This replaces eta = inv(A) @ b without forming the inverse and without
        requiring Cholesky/positive-definiteness.

        Parameters
        ----------
        A_local : ndarray
            Local row slab of the covariance matrix, shape (n_local_rows, n_total).
        b : ndarray
            Full right-hand-side vector, replicated on all ranks.
        tol : float
            Relative convergence tolerance.
        atol : float
            Absolute convergence tolerance.
        restart : int
            GMRES restart length.
        maxiter : int
            Maximum number of GMRES restart cycles in scipy's implementation.

        Returns
        -------
        eta : ndarray
            Full eta vector, replicated on all ranks.
        """
        b = np.asarray(b, dtype=np.float64)
        n = len(b)

        def matvec(x):
            return self._matvec_rowdist(A_local, x)

        Aop = LinearOperator(
            shape=(n, n),
            matvec=matvec,
            dtype=np.float64,
        )

        self._print_zero(" - Starting distributed GMRES")
        self._print_zero(" -- restart =", restart)
        self._print_zero(" -- maxiter =", maxiter)
        self._print_zero(" -- tol     =", tol)

        residual_history = []

        def callback(residual):
            # With callback_type='pr_norm', scipy passes the preconditioned residual norm.
            residual_history.append(float(residual))
            if self.rank == 0:
                print(
                    " --- GMRES iter %i residual %.6e"
                    % (len(residual_history), residual_history[-1]),
                    flush=True,
                )

        # SciPy changed gmres keyword names across versions.
        # Newer scipy uses rtol; older scipy uses tol.
        try:
            eta, info = gmres(
                Aop,
                b,
                rtol=tol,
                atol=atol,
                restart=restart,
                maxiter=maxiter,
                callback=callback,
                callback_type="pr_norm",
            )
        except TypeError:
            eta, info = gmres(
                Aop,
                b,
                tol=tol,
                atol=atol,
                restart=restart,
                maxiter=maxiter,
                callback=callback,
            )

        # Explicit final residual check.
        res = b - self._matvec_rowdist(A_local, eta)

        bnorm = np.linalg.norm(b)
        if bnorm == 0.0:
            relres = np.linalg.norm(res)
        else:
            relres = np.linalg.norm(res) / bnorm

        self._print_zero(" - GMRES info =", info)
        self._print_zero(" - GMRES final relative residual =", relres)

        if info != 0:
            raise np.linalg.LinAlgError(
                "Distributed GMRES did not converge. info=%s, final relres=%.6e"
                % (str(info), relres)
            )

        return np.asarray(eta, dtype=np.float64)
    
    def _extract_type_subcov_local(self, type_id):
        """Extract a type-type covariance block from the row-distributed covariance.

        Returns
        -------
        cond : ndarray
            Global indices of constraints of this type.
        row_pos : ndarray
            Row positions in the reduced type-block owned by this rank.
        A_local : ndarray
            Local rows of C[cond, cond].
        """
        cond = np.where(self.cons_c_type == type_id)[0]

        if len(cond) == 0:
            return cond, None, None

        local_mask = self.cons_c_type[self.cov_rows] == type_id

        row_global = self.cov_rows[local_mask]

        # Position of these global rows inside the reduced type block.
        row_pos = np.searchsorted(cond, row_global)

        A_local = self.cov[local_mask][:, cond].copy()

        return cond, row_pos, A_local


    def _add_sigma_to_type_subcov_local(self, A_local, row_sub, cond, sigma):
        """Return A_local + sigma^2 I for a type-specific distributed block."""
        A = np.asarray(A_local, dtype=np.float64).copy()

        if sigma == 0.0 or len(row_sub) == 0:
            return A

        sigma2 = sigma * sigma

        # cond is sorted global indices. row_sub is a subset of cond.
        diag_pos = np.searchsorted(cond, row_sub)

        for iloc, jloc in enumerate(diag_pos):
            if jloc < len(cond) and cond[jloc] == row_sub[iloc]:
                A[iloc, jloc] += sigma2

        return A


    def _chi2_reduced_type_with_sigma(
        self,
        type_id,
        sigma,
        tol=1e-8,
        restart=50,
        maxiter=500,
    ):
        """Compute reduced chi2 for a type block using distributed GMRES.

        This is the inverse-free equivalent of:
            chi2 = c_type.T @ inv(C_type + sigma^2 I) @ c_type
        """
        cond, row_sub, A0_local = self._extract_type_subcov_local(type_id)

        if len(cond) == 0:
            return np.nan, np.nan, 0

        A_local = self._add_sigma_to_type_subcov_local(
            A0_local,
            row_sub,
            cond,
            sigma,
        )

        b = self.cons_c[cond]

        eta_t = self._solve_eta_rowdist_gmres(
            A_local,
            b,
            tol=tol,
            atol=0.0,
            restart=restart,
            maxiter=maxiter,
        )

        chi2 = float(np.dot(b, eta_t))
        dof = len(b)
        red_chi2 = chi2 / float(dof)

        return chi2, red_chi2, dof


    def _optimise_sigma_NL_type_gmres(
        self,
        type_id,
        name,
        max_sigma,
        target_red_chi2=1.0,
        etol=0.01,
        max_iter=30,
        tol=1e-8,
        restart=50,
        maxiter=500,
    ):
        """Optimise sigma_NL for one constraint type using inverse-free GMRES.

        Uses bisection on sigma_NL to make reduced chi2 approximately one.
        """
        cond = np.where(self.cons_c_type == type_id)[0]

        if len(cond) == 0:
            self._print_zero(" -- No", name, "constraints found")
            return 0.0, True

        self._print_zero(" -- Optimising", name, "dispersion with distributed GMRES")
        self._print_zero(" --- N =", len(cond))

        chi2_lo, red_lo, dof = self._chi2_reduced_type_with_sigma(
            type_id,
            0.0,
            tol=tol,
            restart=restart,
            maxiter=maxiter,
        )

        self._print_zero(
            " --- sigma = %.6e  chi2/dof = %.6f" % (0.0, red_lo)
        )

        # If already below or close to one, no extra dispersion is needed.
        if red_lo <= target_red_chi2 + etol:
            self._print_zero(
                " --- No extra %s dispersion needed; chi2/dof already %.6f"
                % (name, red_lo)
            )
            return 0.0, True

        chi2_hi, red_hi, _ = self._chi2_reduced_type_with_sigma(
            type_id,
            max_sigma,
            tol=tol,
            restart=restart,
            maxiter=maxiter,
        )

        self._print_zero(
            " --- sigma = %.6e  chi2/dof = %.6f" % (max_sigma, red_hi)
        )

        # If max_sigma is not enough, return max_sigma and flag failure.
        if red_hi > target_red_chi2:
            self._print_zero(
                " --- WARNING: max_sigma = %.6e still gives chi2/dof = %.6f"
                % (max_sigma, red_hi)
            )
            return max_sigma, False

        lo = 0.0
        hi = float(max_sigma)

        best_sigma = hi
        best_red = red_hi

        for it in range(max_iter):
            mid = 0.5 * (lo + hi)

            chi2_mid, red_mid, _ = self._chi2_reduced_type_with_sigma(
                type_id,
                mid,
                tol=tol,
                restart=restart,
                maxiter=maxiter,
            )

            self._print_zero(
                " --- iter %02i sigma = %.6e  chi2/dof = %.6f"
                % (it + 1, mid, red_mid)
            )

            best_sigma = mid
            best_red = red_mid

            if abs(red_mid - target_red_chi2) <= etol:
                self._print_zero(
                    " --- success: %s_Sigma_NL = %.6e gives chi2/dof = %.6f"
                    % (name, best_sigma, best_red)
                )
                return best_sigma, True

            # Increasing sigma increases the covariance diagonal and generally
            # decreases chi2.
            if red_mid > target_red_chi2:
                lo = mid
            else:
                hi = mid

        self._print_zero(
            " --- reached max_iter: %s_Sigma_NL = %.6e gives chi2/dof = %.6f"
            % (name, best_sigma, best_red)
        )

        return best_sigma, abs(best_red - target_red_chi2) <= etol
    
    def _matvec_shifted_rowdist(self, A_local, x, sigma2=0.0):
        """Distributed matvec y = (A + sigma2 I) x.

        A_local is row-distributed. x and y are full replicated vectors.
        """
        y = self._matvec_rowdist(A_local, x)

        if sigma2 != 0.0:
            y = y + sigma2 * x

        return y
    
    def _matvec_subrowdist(self, A_local, row_pos, n, x):
        """Distributed matvec for an irregular row-distributed submatrix.

        A_local has local rows of a reduced n x n matrix.
        row_pos gives where each local row belongs in the reduced vector.
        """
        x = np.asarray(x, dtype=np.float64)

        y_local = A_local.dot(x)

        all_rows = self.MPI.collect(row_pos, outlist=True)
        all_vals = self.MPI.collect(y_local, outlist=True)

        if self.rank == 0:
            y = np.zeros(n, dtype=np.float64)

            for rows, vals in zip(all_rows, all_vals):
                y[rows] = vals
        else:
            y = None

        y = self.MPI.broadcast(y)

        return np.asarray(y, dtype=np.float64)
    
    def _matvec_shifted_subrowdist(self, A_local, row_pos, n, x, sigma2=0.0):
        """Compute y = (A + sigma2 I) x for an irregular type subblock."""
        y = self._matvec_subrowdist(A_local, row_pos, n, x)

        if sigma2 != 0.0:
            y = y + sigma2 * x

        return y
    
    def _get_subrowdist_diag(self, A_local, row_pos, n, sigma2=0.0):
        """Return diagonal of an irregular row-distributed submatrix."""
        d_local = np.zeros(n, dtype=np.float64)

        for iloc, ipos in enumerate(row_pos):
            d_local[ipos] = A_local[iloc, ipos] + sigma2

        d = self.MPI.sum(d_local)

        if self.rank != 0:
            d = None

        d = self.MPI.broadcast(d)

        return np.asarray(d, dtype=np.float64)

    def _solve_shifted_symmetric_subrowdist(
        self,
        A_local,
        row_pos,
        b,
        sigma=0.1,
        tol=1e-5,
        maxiter=500,
        x0=None,
    ):
        """Solve (A + sigma^2 I)x = b for a symmetric type subblock.

        Uses CG first, then MINRES, then GMRES as fallback.
        """
        b = np.asarray(b, dtype=np.float64)
        n = len(b)
        sigma2 = float(sigma) * float(sigma)

        def matvec(x):
            return self._matvec_shifted_subrowdist(
                A_local,
                row_pos,
                n,
                x,
                sigma2=sigma2,
            )

        Aop = LinearOperator(
            shape=(n, n),
            matvec=matvec,
            dtype=np.float64,
        )

        # Jacobi preconditioner.
        diag = self._get_subrowdist_diag(
            A_local,
            row_pos,
            n,
            sigma2=sigma2,
        )

        scale = np.median(np.abs(diag))
        if scale <= 0.0 or not np.isfinite(scale):
            scale = np.max(np.abs(diag))
        if scale <= 0.0 or not np.isfinite(scale):
            scale = 1.0

        floor = 1e-12 * scale

        # Positive preconditioner. This is safe for CG/MINRES as a simple SPD M.
        Minv_diag = 1.0 / (np.abs(diag) + floor)

        Mop = LinearOperator(
            shape=(n, n),
            matvec=lambda x: Minv_diag * x,
            dtype=np.float64,
        )

        if x0 is not None:
            x0 = np.asarray(x0, dtype=np.float64)

        # 1. CG attempt.
        try:
            x, info = cg(
                Aop,
                b,
                x0=x0,
                M=Mop,
                rtol=tol,
                atol=0.0,
                maxiter=maxiter,
            )
        except TypeError:
            x, info = cg(
                Aop,
                b,
                x0=x0,
                M=Mop,
                tol=tol,
                atol=0.0,
                maxiter=maxiter,
            )

        if info == 0:
            return x, "cg", info

        self._print_zero(
            " ---- CG failed/stalled with info =", info, "; trying MINRES"
        )

        # 2. MINRES fallback.
        try:
            x, info = minres(
                Aop,
                b,
                x0=x0,
                M=Mop,
                rtol=tol,
                maxiter=maxiter,
            )
        except TypeError:
            x, info = minres(
                Aop,
                b,
                x0=x0,
                M=Mop,
                tol=tol,
                maxiter=maxiter,
            )

        if info == 0:
            return x, "minres", info

        self._print_zero(
            " ---- MINRES failed/stalled with info =", info, "; trying GMRES"
        )

        # 3. Robust fallback. Slower, but should avoid optimiser death.
        try:
            x, info = gmres(
                Aop,
                b,
                x0=x0,
                M=Mop,
                rtol=tol,
                atol=0.0,
                restart=50,
                maxiter=maxiter,
            )
        except TypeError:
            x, info = gmres(
                Aop,
                b,
                x0=x0,
                M=Mop,
                tol=tol,
                atol=0.0,
                restart=50,
                maxiter=maxiter,
            )

        if info == 0:
            return x, "gmres", info

        raise np.linalg.LinAlgError(
            "Shifted type-block solve failed: CG/MINRES/GMRES all failed. "
            "Last info=%s" % str(info)
        )
    
    def _chi2_reduced_type_with_sigma_fast(
        self,
        type_id,
        sigma,
        tol=1e-5,
        maxiter=500,
        x0=None,
    ):
        """Compute chi2/dof for one type block without forming an inverse."""
        cond, row_pos, A_local = self._extract_type_subcov_local(type_id)

        if len(cond) == 0:
            return np.nan, np.nan, 0, None, "none"

        b = self.cons_c[cond]

        eta_t, method_used, info = self._solve_shifted_symmetric_subrowdist(
            A_local,
            row_pos,
            b,
            sigma=sigma,
            tol=tol,
            maxiter=maxiter,
            x0=x0,
        )

        chi2 = float(np.dot(b, eta_t))
        dof = len(b)
        red_chi2 = chi2 / float(dof)

        return chi2, red_chi2, dof, eta_t, method_used
    
    def _optimise_sigma_NL_type_fast(
        self,
        type_id,
        name,
        max_sigma,
        target_red_chi2=1.0,
        etol=0.03,
        max_iter=8,
        solve_tol=1e-4,
        solve_maxiter=200,
    ):
        """Optimise sigma_NL for one constraint type using symmetric solvers.

        Much faster than repeatedly using GMRES.
        """
        cond = np.where(self.cons_c_type == type_id)[0]

        if len(cond) == 0:
            self._print_zero(" -- No", name, "constraints found")
            return 0.0, True

        self._print_zero(" -- Optimising", name, "dispersion")
        self._print_zero(" --- N =", len(cond))

        x0 = None

        try:
            chi2_lo, red_lo, dof, x_lo, method_lo = (
                self._chi2_reduced_type_with_sigma_fast(
                    type_id,
                    sigma=0.0,
                    tol=solve_tol,
                    maxiter=solve_maxiter,
                    x0=None,
                )
            )
        except np.linalg.LinAlgError:
            self._print_zero(
                " --- sigma = 0 solve failed; treating chi2/dof as infinity"
            )
            red_lo = np.inf
            x_lo = None
            method_lo = "failed"

        self._print_zero(
            " --- sigma = %.6e  chi2/dof = %.6f  solver = %s"
            % (0.0, red_lo, method_lo)
        )

        if red_lo <= target_red_chi2 + etol:
            self._print_zero(
                " --- no extra %s dispersion needed" % name
            )
            return 0.0, True

        x0 = x_lo

        chi2_hi, red_hi, _, x_hi, method_hi = self._chi2_reduced_type_with_sigma_fast(
            type_id,
            sigma=max_sigma,
            tol=solve_tol,
            maxiter=solve_maxiter,
            x0=x0,
        )

        self._print_zero(
            " --- sigma = %.6e  chi2/dof = %.6f  solver = %s"
            % (max_sigma, red_hi, method_hi)
        )

        if red_hi > target_red_chi2:
            self._print_zero(
                " --- WARNING: max_sigma = %.6e still gives chi2/dof = %.6f"
                % (max_sigma, red_hi)
            )
            return max_sigma, False

        lo = 0.0
        hi = float(max_sigma)

        best_sigma = hi
        best_red = red_hi
        x0 = x_hi

        for it in range(max_iter):
            mid = 0.5 * (lo + hi)

            chi2_mid, red_mid, _, x_mid, method_mid = (
                self._chi2_reduced_type_with_sigma_fast(
                    type_id,
                    sigma=mid,
                    tol=solve_tol,
                    maxiter=solve_maxiter,
                    x0=x0,
                )
            )

            self._print_zero(
                " --- iter %02i sigma = %.6e  chi2/dof = %.6f  solver = %s"
                % (it + 1, mid, red_mid, method_mid)
            )

            best_sigma = mid
            best_red = red_mid
            x0 = x_mid

            if abs(red_mid - target_red_chi2) <= etol:
                self._print_zero(
                    " --- success: %s_Sigma_NL = %.6e gives chi2/dof = %.6f"
                    % (name, best_sigma, best_red)
                )
                return best_sigma, True

            if red_mid > target_red_chi2:
                lo = mid
            else:
                hi = mid

        self._print_zero(
            " --- reached max_iter: %s_Sigma_NL = %.6e gives chi2/dof = %.6f"
            % (name, best_sigma, best_red)
        )

        return best_sigma, abs(best_red - target_red_chi2) <= etol
    
    def _get_rowdist_diag(self, A_local, row_ind):
        """Return the diagonal of a row-distributed matrix."""
        n = A_local.shape[1]

        d_local = np.zeros(n, dtype=np.float64)

        for iloc, iglob in enumerate(row_ind):
            d_local[iglob] = A_local[iloc, iglob]

        d = self.MPI.sum(d_local)

        if self.rank != 0:
            d = None

        d = self.MPI.broadcast(d)

        return np.asarray(d, dtype=np.float64)


    def _make_rowdist_operator(self, A_local, jitter=0.0):
        """Make LinearOperator for A + jitter I."""
        n = A_local.shape[1]

        def matvec(x):
            y = self._matvec_rowdist(A_local, x)

            if jitter != 0.0:
                y = y + jitter*x

            return y

        return LinearOperator(
            shape=(n, n),
            matvec=matvec,
            dtype=np.float64,
        )


    def _make_rowdist_jacobi_preconditioner(self, A_local, row_ind, jitter=0.0):
        """Make a simple positive Jacobi preconditioner."""
        diag = self._get_rowdist_diag(A_local, row_ind)

        if jitter != 0.0:
            diag = diag + jitter

        scale = np.median(np.abs(diag))

        if not np.isfinite(scale) or scale <= 0.0:
            scale = np.max(np.abs(diag))

        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0

        floor = 1e-12 * scale

        # Use abs(diag) so the preconditioner is positive even if A is indefinite.
        inv_diag = 1.0 / (np.abs(diag) + floor)

        n = len(diag)

        return LinearOperator(
            shape=(n, n),
            matvec=lambda x: inv_diag*x,
            dtype=np.float64,
        )

    def _rowdist_relative_residual(self, A_local, x, b, jitter=0.0):
        """Explicitly compute ||b - (A + jitter I)x|| / ||b||."""
        Ax = self._matvec_rowdist(A_local, x)

        if jitter != 0.0:
            Ax = Ax + jitter*x

        r = b - Ax

        bnorm = np.linalg.norm(b)

        if bnorm == 0.0:
            return np.linalg.norm(r)

        return np.linalg.norm(r) / bnorm
    
    def _solve_eta_rowdist_auto(
        self,
        A_local,
        b,
        row_ind=None,
        tol=1e-7,
        accept_tol=1e-6,
        maxiter=1000,
        gmres_restart=100,
        allow_jitter=True,
    ):
        """Solve A eta = b using several distributed iterative solvers.

        Tries:
            CG          if the matrix behaves SPD
            MINRES      if the matrix is symmetric but indefinite
            BiCGSTAB    if the matrix is mildly nonsymmetric
            GMRES       robust fallback
            jittered GMRES if the system is badly conditioned

        No inverse is formed.
        """
        b = np.asarray(b, dtype=np.float64)

        if row_ind is None:
            row_ind = self.cov_rows

        # Matrix scale used for relative jitter.
        diag = self._get_rowdist_diag(A_local, row_ind)
        scale = np.median(np.abs(diag))

        if not np.isfinite(scale) or scale <= 0.0:
            scale = np.max(np.abs(diag))

        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0

        jitter_factors = [0.0]

        if allow_jitter:
            jitter_factors += [1e-12, 1e-10, 1e-8, 1e-6]

        best_x = None
        best_relres = np.inf
        best_name = None
        best_jitter = 0.0

        for jf in jitter_factors:

            jitter = jf * scale

            if jitter == 0.0:
                self._print_zero(" - Trying covariance solve with no jitter")
            else:
                self._print_zero(
                    " - Trying covariance solve with jitter = %.6e" % jitter
                )

            Aop = self._make_rowdist_operator(A_local, jitter=jitter)
            Mop = self._make_rowdist_jacobi_preconditioner(
                A_local,
                row_ind,
                jitter=jitter,
            )

            solvers = []

            # Only try CG without jitter or with positive jitter.
            solvers.append("cg")

            # MINRES is good for symmetric indefinite matrices.
            solvers.append("minres")

            # BiCGSTAB is cheaper than GMRES and handles mild nonsymmetry.
            solvers.append("bicgstab")

            # GMRES is the robust fallback.
            solvers.append("gmres")

            for solver_name in solvers:

                self._print_zero(" -- Trying", solver_name.upper())

                try:
                    if solver_name == "cg":
                        try:
                            x, info = cg(
                                Aop,
                                b,
                                M=Mop,
                                rtol=tol,
                                atol=0.0,
                                maxiter=maxiter,
                            )
                        except TypeError:
                            x, info = cg(
                                Aop,
                                b,
                                M=Mop,
                                tol=tol,
                                atol=0.0,
                                maxiter=maxiter,
                            )

                    elif solver_name == "minres":
                        try:
                            x, info = minres(
                                Aop,
                                b,
                                M=Mop,
                                rtol=tol,
                                maxiter=maxiter,
                            )
                        except TypeError:
                            x, info = minres(
                                Aop,
                                b,
                                M=Mop,
                                tol=tol,
                                maxiter=maxiter,
                            )

                    elif solver_name == "bicgstab":
                        try:
                            x, info = bicgstab(
                                Aop,
                                b,
                                M=Mop,
                                rtol=tol,
                                atol=0.0,
                                maxiter=maxiter,
                            )
                        except TypeError:
                            x, info = bicgstab(
                                Aop,
                                b,
                                M=Mop,
                                tol=tol,
                                atol=0.0,
                                maxiter=maxiter,
                            )

                    elif solver_name == "gmres":
                        try:
                            x, info = gmres(
                                Aop,
                                b,
                                M=Mop,
                                rtol=tol,
                                atol=0.0,
                                restart=gmres_restart,
                                maxiter=maxiter,
                            )
                        except TypeError:
                            x, info = gmres(
                                Aop,
                                b,
                                M=Mop,
                                tol=tol,
                                atol=0.0,
                                restart=gmres_restart,
                                maxiter=maxiter,
                            )

                except Exception as err:
                    self._print_zero(
                        " --- %s failed with exception: %s"
                        % (solver_name.upper(), str(err))
                    )
                    continue

                relres = self._rowdist_relative_residual(
                    A_local,
                    x,
                    b,
                    jitter=jitter,
                )

                self._print_zero(
                    " --- %s info = %s, relres = %.6e"
                    % (solver_name.upper(), str(info), relres)
                )

                if relres < best_relres:
                    best_x = np.asarray(x, dtype=np.float64)
                    best_relres = relres
                    best_name = solver_name
                    best_jitter = jitter

                # Accept either official convergence or explicit residual convergence.
                if info == 0 and relres <= accept_tol:
                    self._print_zero(
                        " - Accepted %s solve: relres = %.6e, jitter = %.6e"
                        % (solver_name.upper(), relres, jitter)
                    )

                    self.cov_solve_method = solver_name
                    self.cov_solve_relres = relres
                    self.cov_solve_jitter = jitter

                    return np.asarray(x, dtype=np.float64)

                # Sometimes scipy returns nonzero info but the explicit residual is fine.
                if relres <= accept_tol:
                    self._print_zero(
                        " - Accepted %s solve despite info=%s: relres = %.6e, jitter = %.6e"
                        % (solver_name.upper(), str(info), relres, jitter)
                    )

                    self.cov_solve_method = solver_name
                    self.cov_solve_relres = relres
                    self.cov_solve_jitter = jitter

                    return np.asarray(x, dtype=np.float64)

        # Last resort: if the best solution is not terrible, use it.
        relaxed_accept = 1e-4

        if best_x is not None and best_relres <= relaxed_accept:
            self._print_zero(
                " - WARNING: using best relaxed covariance solve."
            )
            self._print_zero(
                " -- method = %s, relres = %.6e, jitter = %.6e"
                % (best_name, best_relres, best_jitter)
            )

            self.cov_solve_method = best_name
            self.cov_solve_relres = best_relres
            self.cov_solve_jitter = best_jitter

            return best_x

        raise np.linalg.LinAlgError(
            "All covariance solvers failed. Best method=%s, best relres=%.6e, jitter=%.6e"
            % (str(best_name), best_relres, best_jitter)
        )

    # Wiener Filtering ----------------------------------------------------

    def _save_cov(self):
        """Save covariance matrix."""
        fname = self._get_fname_prefix() + 'cov.npz'
        np.savez(fname, cov=self.cov, c=self.cons_c, c_type=self.cons_c_type)

    
    def _cov_opt_fast(self):
        """Fast covariance nonlinear-dispersion optimisation."""
        self._print_zero()
        self._print_zero(" - Fast optimisation of nonlinear dispersion errors")

        sigma, success = self._optimise_sigma_NL_type_fast(
            type_id=0,
            name="dens",
            max_sigma=10.0,
            etol=0.03,
            max_iter=8,
            solve_tol=1e-4,
            solve_maxiter=200,
        )

        if success:
            self.constraints["dens_Sigma_NL"] = sigma
        else:
            self.ERROR = True

        io._error_message(self.ERROR, "Density dispersion optimisation failed.", MPI=self.MPI)

        self._break4error()

        sigma, success = self._optimise_sigma_NL_type_fast(
            type_id=1,
            name="psi",
            max_sigma=5.0,
            etol=0.03,
            max_iter=8,
            solve_tol=1e-4,
            solve_maxiter=200,
        )

        if success:
            self.constraints["psi_Sigma_NL"] = sigma
        else:
            self.ERROR = True

        io._error_message(
            self.ERROR,
            "Displacement dispersion optimisation failed.",
            MPI=self.MPI,
        )
        self._break4error()

        sigma, success = self._optimise_sigma_NL_type_fast(
            type_id=2,
            name="vel",
            max_sigma=400.0,
            etol=0.03,
            max_iter=8,
            solve_tol=1e-4,
            solve_maxiter=200,
        )

        if success:
            self.constraints["vel_Sigma_NL"] = sigma
        else:
            self.ERROR = True

        io._error_message(
            self.ERROR,
            "Velocity dispersion optimisation failed.",
            MPI=self.MPI,
        )
        self._break4error()

        self.constraints["dens_Sigma_NL"] = self.MPI.broadcast(
            self.constraints["dens_Sigma_NL"]
        )
        self.constraints["psi_Sigma_NL"] = self.MPI.broadcast(
            self.constraints["psi_Sigma_NL"]
        )
        self.constraints["vel_Sigma_NL"] = self.MPI.broadcast(
            self.constraints["vel_Sigma_NL"]
        )

        self._print_zero()
        self._print_zero(" - Optimised nonlinear dispersions:")
        self._print_zero(" -- dens_Sigma_NL =", self.constraints["dens_Sigma_NL"])
        self._print_zero(" -- psi_Sigma_NL  =", self.constraints["psi_Sigma_NL"])
        self._print_zero(" -- vel_Sigma_NL  =", self.constraints["vel_Sigma_NL"])


    def compute_cov(self):
        """Computes the covariance and eta-vector for the Wiener Filtering."""
        self._print_zero()
        self._print_zero(" Compute covariance and eta-vector")
        self._print_zero(" =================================")
        self._print_zero()

        ncons = len(self.cons_c)

        self.cov_rows = self.MPI.split_array(np.arange(ncons))
        rows = self.cov_rows

        x1, x2 = np.meshgrid(self.cons_x[rows], self.cons_x, indexing='ij')
        y1, y2 = np.meshgrid(self.cons_y[rows], self.cons_y, indexing='ij')
        z1, z2 = np.meshgrid(self.cons_z[rows], self.cons_z, indexing='ij')

        ex1, ex2 = np.meshgrid(self.cons_ex[rows], self.cons_ex, indexing='ij')
        ey1, ey2 = np.meshgrid(self.cons_ey[rows], self.cons_ey, indexing='ij')
        ez1, ez2 = np.meshgrid(self.cons_ez[rows], self.cons_ez, indexing='ij')

        type1, type2 = np.meshgrid(
            self.cons_c_type[rows],
            self.cons_c_type,
            indexing='ij'
        )

        self._print_zero(" - Compute constraint-constraint covariance matrix in parallel")

        if self.constraints["gridcorr"]:
            _cov = theory.get_cc_matrix_fast_grid(
                x1, x2, y1, y2, z1, z2,
                ex1, ex2, ey1, ey2, ez1, ez2, type1, type2,
                self.corr_redshift, self.interp_Hz, self.xi_box, self.zeta_p_box, self.zeta_u_box, 
                self.psixx_pp_box, self.psixy_pp_box, self.psixx_pu_box, self.psixy_pu_box, 
                self.psixx_uu_box, self.psixy_uu_box, self.siminfo["Boxsize"], self.stretch_grid
            )
        else:
            _cov = theory.get_cc_matrix_fast(
                x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2, type1, type2, self.corr_redshift, 
                self.interp_Hz, self.interp_xi, self.interp_zeta_p, self.interp_zeta_u, self.interp_psiR_pp, 
                self.interp_psiT_pp, self.interp_psiR_pu, self.interp_psiT_pu, self.interp_psiR_uu, 
                self.interp_psiT_uu, self.siminfo["Boxsize"]
            )
        
        # if self.constraints["CovOptimise"]:
        #     self._cov_opt()

        self._print_zero(" - Keep constraint-constraint covariance matrix row-distributed")

        ncons = len(self.cons_c)

        self.cov_rows = self.MPI.split_array(np.arange(ncons))
        self.cov = np.asarray(_cov, dtype=np.float64)

        self._add_diag_to_local_rows(
            self.cov,
            self.cov_rows,
            self.cons_c_err**2.
        )
        # Optimise nonlinear dispersions using the base covariance plus measurement errors.
        # This should happen BEFORE adding sigma_NL**2 to self.cov.
        if self.constraints.get("CovOptimise", False):
            self._cov_opt_fast()
        
        sigma_NL = np.ones(len(self.cons_c))

        cond = np.where(self.cons_c_type == 0)[0]
        sigma_NL[cond] = self.constraints["dens_Sigma_NL"]

        cond = np.where(self.cons_c_type == 1)[0]
        sigma_NL[cond] = self.constraints["psi_Sigma_NL"]

        cond = np.where(self.cons_c_type == 2)[0]
        sigma_NL[cond] = self.constraints["vel_Sigma_NL"]

        self._add_diag_to_local_rows(
            self.cov,
            self.cov_rows,
            sigma_NL**2.
        )

        self._print_zero(" - Solving covariance system with automatic distributed solver")

        self.eta = self._solve_eta_rowdist_auto(
            self.cov,
            self.cons_c,
            row_ind=self.cov_rows,
            tol=1e-7,
            accept_tol=1e-6,
            maxiter=1000,
            gmres_restart=100,
            allow_jitter=True,
        )
        
        # Useful final diagnostic.
        chi2 = float(np.dot(self.cons_c, self.eta))
        red_chi2 = chi2 / float(len(self.cons_c))

        self._print_zero(" - Final full chi2      =", chi2)
        self._print_zero(" - Final full chi2/dof  =", red_chi2)

        self.MPI.wait()


    # Real and Fourier Grid functions ------------------------------------------

    def get_grid3D(self):
        """Constructs the 3 dimension grid."""
        if self.x3D is None:
            self._print_zero(" - Construct cartesian grid")
            self.x3D, self.y3D, self.z3D = shift.cart.mpi_grid3D(self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)
            self.x_shape = np.shape(self.x3D)


    def flatten_grid3D(self):
        """Flattens the 3d real space grid."""
        self.x3D = self.x3D.flatten()
        self.y3D = self.y3D.flatten()
        self.z3D = self.z3D.flatten()


    def unflatten_grid3D(self):
        """Unflatten the 3d real space grid."""
        self.x3D = self.x3D.reshape(self.x_shape)
        self.y3D = self.y3D.reshape(self.x_shape)
        self.z3D = self.z3D.reshape(self.x_shape)


    def get_subgrid3D(self):
        """Constructs the 3 dimension grid."""
        if self.sub_x3D is None:
            self._print_zero(" - Construct sub-cartesian grid")
            self.sub_x3D, self.sub_y3D, self.sub_z3D = shift.cart.mpi_grid3D(self.WF["SubBoxsize"], self.WF["SubNgrid"], self.MPI)
            self.sub_x_shape = np.shape(self.sub_x3D)
            center_of_subbox = self.WF["SubBoxsize"]/2.
            self.sub_x3D += self.halfsize - center_of_subbox
            self.sub_y3D += self.halfsize - center_of_subbox
            self.sub_z3D += self.halfsize - center_of_subbox


    def flatten_subgrid3D(self):
        """Flattens the 3d real space sub-grid."""
        self.sub_x3D = self.sub_x3D.flatten()
        self.sub_y3D = self.sub_y3D.flatten()
        self.sub_z3D = self.sub_z3D.flatten()


    def unflatten_subgrid3D(self):
        """Unflatten the 3d real space sub-grid."""
        self.sub_x3D = self.sub_x3D.reshape(self.sub_x_shape)
        self.sub_y3D = self.sub_y3D.reshape(self.sub_x_shape)
        self.sub_z3D = self.sub_z3D.reshape(self.sub_x_shape)


    def get_kgrid3D(self):
        """Constructs the 3 dimensional Fourier Grid"""
        if self.kx3D is None:
            self._print_zero(" - Construct Fourier grid")
            self.kx3D, self.ky3D, self.kz3D = shift.cart.mpi_kgrid3D(self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)
            self.k_shape = np.shape(self.kx3D)


    def flatten_kgrid3D(self):
        """Flattens the 3d Fourier space grid."""
        self.kx3D = self.kx3D.flatten()
        self.ky3D = self.ky3D.flatten()
        self.kz3D = self.kz3D.flatten()


    def unflatten_kgrid3D(self):
        """Unflattens the 3d Fourier space grid."""
        self.kx3D = self.kx3D.reshape(self.k_shape)
        self.ky3D = self.ky3D.reshape(self.k_shape)
        self.kz3D = self.kz3D.reshape(self.k_shape)


    def get_kgrid_mag(self):
        """Returns the Fourier grid magnitudes."""
        return np.sqrt(self.kx3D**2. + self.ky3D**2. + self.kz3D**2.)


    def _MPI_save_xyz(self, suffix="XYZ"):
        """Saves the 3 dimensional grid."""
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


    def _MPI_save_sub_xyz(self, suffix="sub_XYZ"):
        """Saves the 3 dimensional grid."""
        fname_prefix = self._get_fname_prefix()
        fname = fname_prefix + suffix + "_" + str(self.rank) + ".npz"
        if io.isfile(fname) is False:
            check = True
        else:
            data = np.load(fname)
            if data['SubBoxsize'] == self.WF["SubBoxsize"] and data["SubNgrid"] == self.WF["SubNgrid"]:
                check = False
            else:
                check = True
        if check:
            self._print_zero(" - Save sub-box XYZ :", fname_prefix+suffix+"_[0-%i].npz" % (self.MPI.size-1))
            np.savez(fname, SubBoxsize=self.WF["SubBoxsize"], SubNgrid=self.WF["SubNgrid"],
                sub_x3D=self.sub_x3D, sub_y3D=self.sub_y3D, sub_z3D=self.sub_z3D)

    # Wiener Filtering ---------------------------------------------------------

    def _MPI_savez(self, suffix, **kwarg):
        """Generalised MPI save function, in npz format."""
        fname_prefix = self._get_fname_prefix()
        fname = fname_prefix + suffix + "_" + str(self.rank) + ".npz"
        self._print_zero(" - Saving to :", fname_prefix+suffix+"_[0-%i].npz" % (self.MPI.size-1))
        np.savez(fname, **kwarg)


    def _save_WF(self, field, WF):
        """Saves the WF field."""
        self._MPI_save_xyz()
        suffix = "WF_" + field
        self._MPI_savez(suffix, WF=WF)


    def get_WF(self):
        """Computes the WF reconstruction"""
        self._print_zero()
        self._print_zero(" Compute Wiener Filter")
        self._print_zero(" =====================")
        self._print_zero()

        self.get_grid3D()
        self.flatten_grid3D()

        self._print_zero()

        prefix = " ---- "

        if self.WF["Field"] == "dens":
            self._print_zero(" - Computing Wiener Filter density")
            typei = 0
            exi, eyi, ezi = 1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)
        elif self.WF["Field"] == "psi_x":
            self._print_zero(" - Computing Wiener Filter displacment in x")
            typei = 1
            exi, eyi, ezi = 1., 0., 0.
        elif self.WF["Field"] == "psi_y":
            self._print_zero(" - Computing Wiener Filter displacment in y")
            typei = 1
            exi, eyi, ezi = 0., 1., 0.
        elif self.WF["Field"] == "psi_z":
            self._print_zero(" - Computing Wiener Filter displacment in z")
            typei = 1
            exi, eyi, ezi = 0., 0., 1.
        elif self.WF["Field"] == "psi_r":
            self._print_zero(" - Computing Wiener Filter displacment in r")
            typei = 1
            exi = self.x3D - self.halfsize
            eyi = self.y3D - self.halfsize
            ezi = self.z3D - self.halfsize
            _r = np.sqrt(exi**2. + eyi**2. + ezi**2.)
            exi /= _r
            eyi /= _r
            ezi /= _r
        elif self.WF["Field"] == "vel_x":
            self._print_zero(" - Computing Wiener Filter velocity in x")
            typei = 2
            exi, eyi, ezi = 1., 0., 0.
        elif self.WF["Field"] == "vel_y":
            self._print_zero(" - Computing Wiener Filter velocity in y")
            typei = 2
            exi, eyi, ezi = 0., 1., 0.
        elif self.WF["Field"] == "vel_z":
            self._print_zero(" - Computing Wiener Filter velocity in z")
            typei = 2
            exi, eyi, ezi = 0., 0., 1.
        elif self.WF["Field"] == "vel_r":
            self._print_zero(" - Computing Wiener Filter velocity in r")
            typei = 2
            exi = self.x3D - self.halfsize
            eyi = self.y3D - self.halfsize
            ezi = self.z3D - self.halfsize
            _r = np.sqrt(exi**2. + eyi**2. + ezi**2.)
            exi /= _r
            eyi /= _r
            ezi /= _r

        if self.what2run["WF"]:
            if self.constraints["gridcorr"]:
                WF = theory.get_corr_dot_eta_fast_grid(
                    self.x3D, self.cons_x, self.y3D, self.cons_y, self.z3D, self.cons_z, 
                    exi, self.cons_ex, eyi, self.cons_ey, ezi, self.cons_ez,
                    typei, self.cons_c_type, self.corr_redshift, self.interp_Hz, 
                    self.xi_box, self.zeta_p_box, self.zeta_u_box, self.psixx_pp_box, self.psixy_pp_box,
                    self.psixx_pu_box, self.psixy_pu_box, self.psixx_uu_box, self.psixy_uu_box, self.eta,
                    self.siminfo["Boxsize"], self._lenpro+2, prefix, self.stretch_grid, mpi_rank=self.MPI.rank,
                )
            else:
                WF = theory.get_corr_dot_eta_fast(
                    self.x3D, self.cons_x, self.y3D, self.cons_y, self.z3D, self.cons_z, 
                    exi, self.cons_ex, eyi, self.cons_ey, ezi, self.cons_ez,
                    typei, self.cons_c_type, self.corr_redshift, self.interp_Hz, self.interp_xi,
                    self.interp_zeta_p, self.interp_zeta_u, self.interp_psiR_pp, self.interp_psiT_pp,
                    self.interp_psiR_pu, self.interp_psiT_pu, self.interp_psiR_uu, self.interp_psiT_uu,
                    self.eta, self.siminfo["Boxsize"], self._lenpro+2, prefix, mpi_rank=self.MPI.rank, 
                    minlogr=-2
                )

        self.unflatten_grid3D()

        if self.what2run["WF"]:
            WF = WF.reshape(self.x_shape)
            self._save_WF(self.WF["Field"], WF)

        self._print_zero()

        if self.WF["Convert"] is not None:

            dens = WF

            z0 = self.constraints["z_eff"]

            self._print_zero(" - Computing displacement field Psi from density")

            psi_x, psi_y, psi_z = self.dens2psi(dens)

            self._print_zero(" - Computing velocity field from displacement field Psi")

            if self.WF["Convert"] == 'psi':
                self._MPI_savez('WF_dens2psi', psi_x=psi_x, psi_y=psi_y, psi_z=psi_z)
            else:
                vel_x, vel_y, vel_z = self.psi2vel(z0, psi_x, psi_y, psi_z)
                self._MPI_savez('WF_dens2vel', vel_x=vel_x, vel_y=vel_y, vel_z=vel_z)

        if self.what2run["RZA"]:
            self.dens_WF = WF
    

    # FFT related functions ----------------------------------------------------

    def complex_zeros(self, shape):
        """Construct complex zeros."""
        return np.zeros(shape) + 1j*np.zeros(shape)


    # Density to displacement/velocity functions -------------------------------

    def dens2psi(self, dens):
        """Conversion from density to displacement fields along each cartesian
        axes."""
        self.get_grid3D()
        self.get_kgrid3D()
        kmag = self.get_kgrid_mag()
        densk = shift.cart.mpi_fft3D(dens, self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)
        psi_kx = self.complex_zeros(self.k_shape)
        psi_ky = self.complex_zeros(self.k_shape)
        psi_kz = self.complex_zeros(self.k_shape)
        cond = np.where(kmag != 0.)
        psi_kx[cond] = densk[cond] * 1j * self.kx3D[cond]/(kmag[cond]**2.)
        psi_ky[cond] = densk[cond] * 1j * self.ky3D[cond]/(kmag[cond]**2.)
        psi_kz[cond] = densk[cond] * 1j * self.kz3D[cond]/(kmag[cond]**2.)
        psi_x = shift.cart.mpi_ifft3D(psi_kx, self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)
        psi_y = shift.cart.mpi_ifft3D(psi_ky, self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)
        psi_z = shift.cart.mpi_ifft3D(psi_kz, self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)
        return psi_x, psi_y, psi_z


    def psi2vel(self, redshift, psi_x, psi_y, psi_z):
        """Conversion from displacement to velocity fields along each cartesian
        axes."""
        z0 = redshift
        Hz = self.interp_Hz(z0)

        adot = theory.z2a(z0)*Hz

        if self.cosmo["ScaleDepGrowth"]:
            self.get_kgrid3D()
            kmag = self.get_kgrid_mag()

            vel_kx = shift.cart.mpi_fft3D(psi_x, self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)
            vel_ky = shift.cart.mpi_fft3D(psi_y, self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)
            vel_kz = shift.cart.mpi_fft3D(psi_z, self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)

            cond = np.where(kmag != 0.)
            fk = self._get_growth_f(z0, kmag=kmag[cond])
            vel_kx[cond] *= adot*fk
            vel_ky[cond] *= adot*fk
            vel_kz[cond] *= adot*fk

            vel_x = shift.cart.mpi_ifft3D(vel_kx, self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)
            vel_y = shift.cart.mpi_ifft3D(vel_ky, self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)
            vel_z = shift.cart.mpi_ifft3D(vel_kz, self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)

        else:
            fz = self._get_growth_f(z0)
            vel_x = adot*fz*psi_x
            vel_y = adot*fz*psi_y
            vel_z = adot*fz*psi_z

        return vel_x, vel_y, vel_z

    # Including buffer regions for distributed grids ---------------------------

    def _add_buffer_in_x(self, f):
        """Add buffer regions along the x-axes which is the axes in which
        parallelisation is performed."""
        f_send_down = self.MPI.send_down(f[0])
        f_send_up = self.MPI.send_up(f[-1])
        fnew = np.concatenate([np.array([f_send_up]), f, np.array([f_send_down])])
        return fnew


    def _get_buffer_range(self):
        """Returns the buffer region range."""
        dx = self.siminfo["Boxsize"]/self.siminfo["Ngrid"]
        xmin = np.min(self.x3D) - dx/2. - dx
        xmax = np.max(self.x3D) + dx/2. + dx
        return xmin, xmax


    def _unbuffer_in_x(self, f):
        """Removes buffer regions."""
        fnew = f[1:-1]
        return fnew

    # Apply the Reverse Zel'dovich approximation -------------------------------

    def get_RZA(self):
        """Apply the reverse Zel'dovich approximation."""
        self._print_zero()
        self._print_zero(" Apply Reverse Zeldovich Approximation")
        self._print_zero(" =====================================")
        self._print_zero()

        self._print_zero(" - Converting WF density to displacement Psi")
        self._print_zero()

        if self.dens_WF is None:
            self.ERROR = True
        io._error_message(self.ERROR, "dens_WF is None, thus cannot compute RZA from this.")
        self._break4error()

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

            cons_x = np.asarray(cons_x, dtype=np.float64)
            cons_y = np.asarray(cons_y, dtype=np.float64)
            cons_z = np.asarray(cons_z, dtype=np.float64)

            cons_ex = np.asarray(cons_ex, dtype=np.float64)
            cons_ey = np.asarray(cons_ey, dtype=np.float64)
            cons_ez = np.asarray(cons_ez, dtype=np.float64)

            cons_c = np.asarray(cons_c, dtype=np.float64)
            cons_c_err = np.asarray(cons_c_err, dtype=np.float64)

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

        if self.RZA["Method"] == 2:
            # This assumes Method II of https://theses.hal.science/tel-01127294/document see page 121
            cons_rza_x = cons_x - cons_psi_x
            cons_rza_y = cons_y - cons_psi_y
            cons_rza_z = cons_z - cons_psi_z
            cons_rza_ex = np.copy(cons_ex)
            cons_rza_ey = np.copy(cons_ey)
            cons_rza_ez = np.copy(cons_ez)
            cons_rza_c = np.copy(cons_c)
            cons_rza_c_err = np.copy(cons_c_err)
            cons_rza_c_type = 2*np.ones(len(cons_c))

        elif self.RZA["Method"] == 3:
            # See above paper for Method III
            cons_rza_x = cons_x - cons_psi_x
            cons_rza_y = cons_y - cons_psi_y
            cons_rza_z = cons_z - cons_psi_z
            cons_rza_ex = np.copy(cons_ex)
            cons_rza_ey = np.copy(cons_ey)
            cons_rza_ez = np.copy(cons_ez)
            cons_rza_c = np.copy(cons_c)
            cons_rza_c_err = np.zeros(len(cons_c_err))
            cons_rza_c = np.sqrt(cons_rza_x**2. + cons_rza_y**2. + cons_rza_z**2.)
            cons_rza_ex = cons_rza_x / cons_rza_c
            cons_rza_ey = cons_rza_y / cons_rza_c
            cons_rza_ez = cons_rza_z / cons_rza_c
            cons_rza_c_type = 1*np.ones(len(cons_c))

        self.cons_x = self.MPI.collect_noNone(cons_rza_x)
        self.cons_y = self.MPI.collect_noNone(cons_rza_y)
        self.cons_z = self.MPI.collect_noNone(cons_rza_z)

        self.cons_ex = self.MPI.collect_noNone(cons_rza_ex)
        self.cons_ey = self.MPI.collect_noNone(cons_rza_ey)
        self.cons_ez = self.MPI.collect_noNone(cons_rza_ez)

        self.cons_c = self.MPI.collect_noNone(cons_rza_c)
        self.cons_c_err = self.MPI.collect_noNone(cons_rza_c_err)
        self.cons_c_type = self.MPI.collect_noNone(cons_rza_c_type)

        self.cons_x = self.MPI.broadcast(self.cons_x)
        self.cons_y = self.MPI.broadcast(self.cons_y)
        self.cons_z = self.MPI.broadcast(self.cons_z)

        self.cons_x %= self.siminfo["Boxsize"]
        self.cons_y %= self.siminfo["Boxsize"]
        self.cons_z %= self.siminfo["Boxsize"]

        self.cons_ex = self.MPI.broadcast(self.cons_ex)
        self.cons_ey = self.MPI.broadcast(self.cons_ey)
        self.cons_ez = self.MPI.broadcast(self.cons_ez)

        self.cons_c = self.MPI.broadcast(self.cons_c)
        self.cons_c_err = self.MPI.broadcast(self.cons_c_err)
        self.cons_c_type = self.MPI.broadcast(self.cons_c_type)

        fname = self._get_fname_prefix() + 'rza.npz'
        self._print_zero(" - Saving RZA constraints to: %s" % fname)

        if self.rank == 0:
            io._save_constraints_npz(fname, self.cons_x-self.halfsize, self.cons_y-self.halfsize,
                self.cons_z-self.halfsize, self.cons_ex, self.cons_ey, self.cons_ez,
                self.cons_c, self.cons_c_err, self.cons_c_type)

    # Random realisations ------------------------------------------------------

    def save_WN(self, WN):
        """Saves the whitenoise fields."""
        self._MPI_save_xyz()
        suffix = "WN"
        self._MPI_savez(suffix, WN=WN)


    def save_dens(self, suffix):
        """Save density field."""
        self._MPI_save_xyz()
        self._MPI_savez(suffix, dens=self.dens)


    def get_RR(self):
        """Produces a random Gaussian field."""
        self._print_zero()
        self._print_zero(" Construct Random Realisation")
        self._print_zero(" ============================")
        self._print_zero()

        self.get_grid3D()
        self.get_kgrid3D()
        kmag = self.get_kgrid_mag()
        
        if self.ICs["Seed"] is not None:
            self._print_zero(" - Construct white noise field with seed %i" % self.ICs["Seed"])
            WN = field.get_white_noise_3D(self.ICs["Seed"], self.siminfo["Ngrid"], MPI=self.MPI)
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

        WN_k = shift.cart.mpi_fft3D(WN, self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)

        self._print_zero(" - Colour white noise field to get density field")

        dx = self.siminfo["Boxsize"]/self.siminfo["Ngrid"]
        dens_RR_k = field.color_white_noise(WN_k, dx, kmag, self.interp_pk, mode='3D')

        self._print_zero(" - iFFT density field")

        self.dens = shift.cart.mpi_ifft3D(dens_RR_k, self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)

        self.save_dens("RR")


    def dens_at_z(self, dens, redshift, redshift_current=0.):
        """Scales a density with the growth function from a current redshift to
        some desired redshift.

        Parameters
        ----------
        dens : array
            Density on a grid.
        redshift : float
            Desired redshift.
        redshift_current : float
            Current redshift of the density input.
        """
        z0 = redshift
        z1 = redshift_current
        if self.cosmo["ScaleDepGrowth"]:
            self.get_kgrid3D()
            kmag = self.get_kgrid_mag()
            dens_k = shift.cart.mpi_fft3D(dens, self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)
            cond = np.where(kmag != 0.)
            Dk = self._get_growth_D(z0, kmag=kmag[cond])
            Dk0 = self._get_growth_D(z1, kmag=kmag[cond])
            dens_k[cond] = (Dk/Dk0)*dens_k[cond]
            densz = shift.cart.mpi_ifft3D(dens_k, self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)
        else:
            Dz = self._get_growth_D(z0)
            Dz0 = self._get_growth_D(z1)
            densz = (Dz/Dz0)*dens
        return densz
    

    def compute_eta_CR(self):
        """Computes the eta vector for a constrained realisation."""
        self._print_zero()
        self._print_zero(" Compute eta_CR-vector")
        self._print_zero(" =====================")
        self._print_zero()

        self._print_zero(" - Compute eta_CR vector with distributed GMRES")

        rhs = self.cons_c - self.cons_c_RR

        self.eta_CR = self._solve_eta_rowdist_gmres(
            self.cov,
            rhs,
            tol=1e-8,
            atol=0.0,
            restart=50,
            maxiter=500,
        )

        self.MPI.wait()


    def prep_CR(self):
        """Prepares theory, etc for constrained realisation calculation."""
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

            cons_x = np.asarray(cons_x, dtype=np.float64)
            cons_y = np.asarray(cons_y, dtype=np.float64)
            cons_z = np.asarray(cons_z, dtype=np.float64)

            cons_ex = np.asarray(cons_ex, dtype=np.float64)
            cons_ey = np.asarray(cons_ey, dtype=np.float64)
            cons_ez = np.asarray(cons_ez, dtype=np.float64)

            cons_c = np.asarray(cons_c, dtype=np.float64)
            cons_c_err = np.asarray(cons_c_err, dtype=np.float64)

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

        self.compute_cov()

        self.compute_eta_CR()


    def get_CR(self):
        """Computes a constrained realisation field."""
        self._print_zero()
        self._print_zero(" Construct Constrained Realisation")
        self._print_zero(" =================================")
        self._print_zero()

        z0 = self.constraints["z_eff"]

        self.get_grid3D()
        self.flatten_grid3D()

        self.dens = self.dens.flatten()

        self._print_zero(" - Computing Constrained Realisation density")

        prefix = " ---- "

        typei = 0
        exi, eyi, ezi = 1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)

        if self.constraints["gridcorr"]:
            self.dens += theory.get_corr_dot_eta_fast_grid(
                self.x3D, self.cons_x, self.y3D, self.cons_y, self.z3D, self.cons_z,
                exi, self.cons_ex, eyi, self.cons_ey, ezi, self.cons_ez, typei, self.cons_c_type, self.corr_redshift, 
                self.interp_Hz, self.xi_box, self.zeta_p_box, self.zeta_u_box, self.psixx_pp_box, 
                self.psixy_pp_box, self.psixx_pu_box, self.psixy_pu_box, self.psixx_uu_box, self.psixy_uu_box,
                self.eta_CR, self.siminfo["Boxsize"], self._lenpro+2, prefix, self.stretch_grid, mpi_rank=self.MPI.rank
            )
        else:
            self.dens += theory.get_corr_dot_eta_fast(
                self.x3D, self.cons_x, self.y3D, self.cons_y, self.z3D, self.cons_z, exi, self.cons_ex, 
                eyi, self.cons_ey, ezi, self.cons_ez, typei, self.cons_c_type, self.corr_redshift, 
                self.interp_Hz, self.interp_xi, self.interp_zeta_p, self.interp_zeta_u, self.interp_psiR_pp, 
                self.interp_psiT_pp, self.interp_psiR_pu, self.interp_psiT_pu, self.interp_psiR_uu, 
                self.interp_psiT_uu, self.eta_CR, self.siminfo["Boxsize"], lenpro=self._lenpro+2, 
                prefix=prefix, mpi_rank=self.MPI.rank, minlogr=-2
            )
        self.unflatten_grid3D()
        self.dens = self.dens.reshape(self.x_shape)

        self._print_zero()
        self.save_dens("CR")


    # Initial condition functions ----------------------------------------------

    def get_particle_mass(self):
        """Determines the particle mass."""
        G_const = 6.6743e-11
        part_mass = 3.*self.cosmo["Omega_m"]*self.siminfo["Boxsize"]**3.
        part_mass /= 8.*np.pi*G_const*self.siminfo["Ngrid"]**3.
        part_mass *= 3.0857e2/1.9891
        part_mass /= 1e10
        return part_mass


    def get_IC(self):
        """Produces initial conditions."""
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

        a_ic = theory.z2a(z0)

        vel_x /= np.sqrt(a_ic)
        vel_y /= np.sqrt(a_ic)
        vel_z /= np.sqrt(a_ic)

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

        fname = self._get_fname_prefix() + 'IC.%i' % self.rank

        self._print_zero()
        self._print_zero(" - Saving ICs in Gadget format to %s[0-%i]"%(fname[:-1], self.MPI.size-1))

        io.save_gadget(fname, header, pos, vel, ic_format=self.ICs['gadget_format'], single=True, id_offset=part_id_offsets[self.rank])


    # Main pipeline running ----------------------------------------------------

    def run(self, yaml_fname):
        """Run MIMIC."""
        self.start()
        self.read_paramfile(yaml_fname)
        # Theory
        self.time["Prep_Start"] = time.time()
        self.prep()
        self.compute_cov()
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
