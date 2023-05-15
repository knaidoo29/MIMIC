import numpy as np

from ..utils import printout as printout
from ..utils import printspace as printspace


def print_start(MPI, verbose):
    if MPI.rank == 0:
        printout("=====================================================================", 0, verbose)
        printout("=                     __  _________  _____________                  =", 0, verbose)
        printout("=                    /  |/  /  _/  |/  /  _/ ____/                  =", 0, verbose)
        printout("=                   / /|_/ // // /|_/ // // /                       =", 0, verbose)
        printout("=                  / /  / // // /  / // // /___                     =", 0, verbose)
        printout("=                 /_/  /_/___/_/  /_/___/\____/                     =", 0, verbose)
        printout("=                                                                   =", 0, verbose)
        printout("=====================================================================", 0, verbose)
        printout("=-- Model Independent constrained cosMological Initial Conditions --=", 0, verbose)
        printout("=====================================================================", 0, verbose)
        printspace(verbose)
        printout("> INITIALISING -----------------------------------------------------<", 0, verbose)
        printspace(verbose)
        printout("> Running MPI with " + str(MPI.size) + " processors", 0, verbose)


def print_params(params, MPI, verbose):
    if MPI.rank == 0:
        printspace(verbose)
        printout("> Cosmology", 0, verbose)
        printout("H0             = "+str(params["Cosmology"]["H0"]), 1, verbose)
        printout("Omega_m        = "+str(params["Cosmology"]["Omega_m"]), 1, verbose)
        printout("PowerSpecFile  = "+params["Cosmology"]["PowerSpecFile"], 1, verbose)
        printout("ScaleDepGrowth = "+str(params["Cosmology"]["ScaleDepGrowth"]), 1, verbose)
        printout("GrowthFile     = "+str(params["Cosmology"]["GrowthFile"]), 1, verbose)

        if params["Constraints"] is not None:
            printspace(verbose)
            printout("> Constraints", 0, verbose)
            printout("File           = "+params["Constraints"]["File"], 1, verbose)
            printout("z_eff          = "+str(params["Constraints"]["z_eff"]), 1, verbose)
            printout("Rg             = "+str(params["Constraints"]["Rg"]), 1, verbose)
            printout("CorrFile       = "+str(params["Constraints"]["CorrFile"]), 1, verbose)
            printout("CovFile        = "+str(params["Constraints"]["CovFile"]), 1, verbose)
            printout("Sigma_NR       = "+str(params["Constraints"]["Sigma_NR"]), 1, verbose)
            printout("Type           = "+str(params["Constraints"]["Type"]), 1, verbose)

        if params["Randoms"] is not None:
            printspace(verbose)
            printout("> Randoms", 0, verbose)
            printout("Seed           = "+str(params["Randoms"]["Seed"]), 1, verbose)

        printspace(verbose)
        printout("> Siminfo", 0, verbose)
        printout("Simname        = "+str(params["Siminfo"]["Simname"]), 1, verbose)
        printout("Boxsize        = "+str(params["Siminfo"]["Boxsize"]), 1, verbose)
        printout("Ngrid          = "+str(params["Siminfo"]["Ngrid"]), 1, verbose)

        if params['Enhance'] is not None:
            printspace(verbose)
            printout("> Enhance", 0, verbose)
            printout("Seed           = "+str(params["Enhance"]["Seed"]), 1, verbose)
            printout("Ngrid          = "+str(params["Enhance"]["Ngrid"]), 1, verbose)

        if params['ICs'] is not None:
            printspace(verbose)
            printout("> ICs", 0, verbose)
            printout("z_init         = "+str(params["ICs"]["z_init"]), 1, verbose)
            printout("Single         = "+str(params["ICs"]["Single"]), 1, verbose)

        printspace(verbose)
        printout("> Run", 0, verbose)
        printout("WF_Den         = "+str(params["Run"]["WF_Den"]), 1, verbose)
        printout("WF_Psi         = "+str(params["Run"]["WF_Psi"]), 1, verbose)
        printout("WF_Vel         = "+str(params["Run"]["WF_Vel"]), 1, verbose)
        printout("RZA            = "+str(params["Run"]["RZA"]), 1, verbose)
        printout("CR             = "+str(params["Run"]["CR"]), 1, verbose)
        printout("Enhance        = "+str(params["Run"]["Enhance"]), 1, verbose)
        printout("IC             = "+str(params["Run"]["IC"]), 1, verbose)

        printspace(verbose)
        printout("> Check setup", 0, verbose)
        if params["Constraints"] is None:
            params["Run"]["WF_Den"] = False
            params["Run"]["WF_Psi"] = False
            params["Run"]["WF_Vel"] = False
            params["Run"]["RZA"] = False
            params["Run"]["CR"] = False
            printout("Since no constraints provided:", 1, verbose)
            printout("WF_Den         = "+str(params["Run"]["WF_Den"]), 2, verbose)
            printout("WF_Psi         = "+str(params["Run"]["WF_Psi"]), 1, verbose)
            printout("WF_Vel         = "+str(params["Run"]["WF_Vel"]), 2, verbose)
            printout("RZA            = "+str(params["Run"]["RZA"]), 2, verbose)
            printout("CR             = "+str(params["Run"]["CR"]), 2, verbose)

        if params["Run"]["WF_Psi"] or params["Run"]["WF_Vel"]:
            if params["Run"]["WF_Den"] is False:
                params["Run"]["WF_Den"] = True
                printout("Since WF_Psi = True or WF_Vel = True and is computed from WF_Den then:", 1, verbose)
                printout("WF_Den         = "+str(params["Run"]["WF_Den"]), 2, verbose)

        if params["Run"]["RZA"]:
            if params["Run"]["WF_Den"] is False:
                params["Run"]["WF_Den"] = True
                printout("Since RZA = True then:", 1, verbose)
                printout("WF_Den         = "+str(params["Run"]["WF_Den"]), 2, verbose)


def start_constraints(MPI, verbose):
    if MPI.rank == 0:
        printspace(verbose)
        printout("> COMPUTE CORRELATORS ----------------------------------------------<", 0, verbose)
