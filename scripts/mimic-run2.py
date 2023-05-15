import sys
import os.path
import numpy as np
from scipy.interpolate import interp1d

import shift
import fiesta
import mimic

from mimic.utils import printout, printspace
import mpiutils

MPI = mpiutils.MPI()

verbose = True

# Read parameter filename
yaml_fname = str(sys.argv[1])

# Print MIMIC start
mimic.main.print_start(MPI, verbose)

# Real parameter file
params, ERROR = mimic.main.read_paramfile(yaml_fname, MPI, verbose)

# print parameters
if ERROR is False:
    mimic.main.print_params(params, MPI, verbose)

if ERROR is False and params["Constraints"] is not None:
    mimic.main.start_constraints(MPI, verbose)

    if MPI.rank == 0:
        printspace(verbose)
        printout("Load PowerSpecFile", 1, verbose)
    data = np.load(params["Cosmology"]["PowerSpecFile"])
    kh, pk = data['kh'], data['pk']

    if MPI.rank == 0:
        printout("Create P(k) interpolator", 1, verbose)

    kmin, kmax = kh.min(), kh.max()
    interp_pk = interp1d(kh, pk, kind='cubic', bounds_error=False, fill_value=0.)

    if MPI.rank == 0:
        printspace(verbose)
        printout("Load GrowthFile", 1, verbose)

    if params["Cosmology"]["ScaleDepGrowth"]:
        data = np.load(params["Cosmology"]["GrowthFile"])
        growth_z, growth_Hz, growth_kh, growth_Dzk, growth_fzk = data['z'], data['Hz'], data['kh'], data['Dzk'], data['fzk']
    else:
        data = np.load(params["Cosmology"]["GrowthFile"])
        growth_z, growth_Hz, growth_Dz, growth_fz = data['z'], data['Hz'], data['Dz'], data['fz']

    if MPI.rank == 0:
        printout("Create H(z) interpolator", 1, verbose)
    interp_Hz = interp1d(growth_z, growth_Hz/(params["Cosmology"]["H0"]*1e-2), kind='cubic')

    if MPI.rank == 0:
        printout("Create D(z, k) interpolator", 1, verbose)

    if params["Cosmology"]["ScaleDepGrowth"]:
        def interp_Dzk_2_Dk(redshift, kind='cubic'):
            growth_Dk = np.zeros(len(growth_kh))
            for i in range(0, len(growth_kh)):
                interp_D = interp1d(growth_z, growth_Dzk[:, i], kind=kind)
                growth_Dk[i] = interp_D(redshift)
            interp_Dk = interp1d(growth_kh, growth_Dk, kind=kind, bounds_error=False)
            return interp_Dk
    else:
        interp_Dz = interp1d(growth_z, growth_Dz, kind='cubic')

    if MPI.rank == 0:
        printout("Create f(z, k) interpolator", 1, verbose)

    if params["Cosmology"]["ScaleDepGrowth"]:
        def interp_fzk_2_fk(redshift, kind='cubic'):
            growth_fk = np.zeros(len(growth_kh))
            for i in range(0, len(growth_kh)):
                interp_f = interp1d(growth_z, growth_fzk[:, i], kind=kind)
                growth_fk[i] = interp_f(redshift)
            interp_fk = interp1d(growth_kh, growth_fk, kind=kind, bounds_error=False)
            return interp_fk
    else:
        interp_fz = interp1d(growth_z, growth_fz, kind='cubic')

    ## UP TO HERE
    if MPI.rank == 0:
        printspace(verbose)
        printout("Load constraints", 1, verbose)

    data = np.load(params["Constraints"]["File"])
    cons = {
    'x': data['x'], 'y': data['y'], 'z': data['z'],
    'ex': data['ex'], 'ey': data['ey'], 'ez': data['ez'],
    'u': data['u'], 'u_err': data['u_err']
    }

    if MPI.rank == 0:
        printout("Move constraint to the center of simulation and remove outside boundaries", 1, verbose)

    halfsize = params["Siminfo"]["Boxsize"]/2.
    cons['x'] += halfsize
    cons['y'] += halfsize
    cons['z'] += halfsize

    factor = 1.#0.8
    r = np.sqrt((cons['x']-halfsize)**2. + (cons['y']-halfsize)**2. + (cons['z']-halfsize)**2.)
    cond = np.where(r < factor*halfsize)[0]

    if MPI.rank == 0:
        printout('Keeping '+str(len(cond))+" out of " + str(len(r)), 2, verbose)

    cons['x'], cons['y'], cons['z'] = cons['x'][cond], cons['y'][cond], cons['z'][cond]
    cons['ex'], cons['ey'], cons['ez'] = cons['ex'][cond], cons['ey'][cond], cons['ez'][cond]
    cons['u'], cons['u_err'] = cons['u'][cond], cons['u_err'][cond]

    if MPI.rank == 0:
        printout("Check constraint unit vectors are normalised", 1, verbose)
    # KN Note: unparalleled for the moment but might want to spread this out
    # for larger data sets.
    norm = np.sqrt(cons['ex']**2. + cons['ey']**2. + cons['ez']**2. )
    cons['ex'] /= norm
    cons['ey'] /= norm
    cons['ez'] /= norm

    if MPI.rank == 0:
        printspace(verbose)
        printout("Compute correlation functions", 1, verbose)

    r = np.logspace(-2., np.log10(np.sqrt(3.)*params["Siminfo"]["Boxsize"]), 500)
    sim_kmin = shift.cart.get_kf(params["Siminfo"]["Boxsize"])
    sim_kmax = shift.cart.get_kn(params["Siminfo"]["Boxsize"], params["Siminfo"]["Ngrid"])

    z0 = params["Constraints"]["z_eff"]

    if params["Cosmology"]["ScaleDepGrowth"]:
        interp_Dk = interp_Dzk_2_Dk(z0)
        interp_fk = interp_fzk_2_fk(z0)
        Dz2 = interp_Dk(kh)**2
        fz0 = interp_fk(kh)
    else:
        Dz2 = interp_Dz(z0)**2
        fz0 = interp_fz(z0)

    Rg = params["Constraints"]["Rg"]
    #if Rg is None:
    #    Rg = 2.*np.pi/sim_kmax

    if MPI.size >= 4:
        if MPI.rank == 0:
            xi = mimic.theory.pk2xi(r, kh, Dz2*pk, kmin=sim_kmin, kmax=sim_kmax, kfactor=100,
                                    kbinsmin=int(1e4), kbinsmax=int(1e6), Rg=Rg)
            MPI.send(xi, to_rank=None, tag=11)
            zeta = MPI.recv(1, tag=12)
            psiR = MPI.recv(2, tag=13)
            psiT = MPI.recv(3, tag=14)
        elif MPI.rank == 1:
            zeta = mimic.theory.pk2zeta(r, kh, Dz2*pk, fk=fz0, kmin=sim_kmin, kmax=sim_kmax,
                                        kfactor=100, kbinsmin=int(1e4),
                                        kbinsmax=int(1e6), Rg=Rg, cons_type=params["Constraints"]["Type"])
            MPI.send(zeta, to_rank=None, tag=12)
            xi = MPI.recv(0, tag=11)
            psiR = MPI.recv(2, tag=13)
            psiT = MPI.recv(3, tag=14)
        elif MPI.rank == 2:
            psiR = mimic.theory.pk2psiR(r, kh, Dz2*pk, fk=fz0, kmin=sim_kmin, kmax=sim_kmax,
                                        kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6),
                                        Rg=Rg, cons_type=params["Constraints"]["Type"])
            MPI.send(psiR, to_rank=None, tag=13)
            xi = MPI.recv(0, tag=11)
            zeta = MPI.recv(1, tag=12)
            psiT = MPI.recv(3, tag=14)
        elif MPI.rank == 3:
            psiT = mimic.theory.pk2psiT(r, kh, Dz2*pk, fk=fz0, kmin=sim_kmin, kmax=sim_kmax,
                                        kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6),
                                        Rg=Rg, cons_type=params["Constraints"]["Type"])
            MPI.send(psiT, to_rank=None, tag=14)
            xi = MPI.recv(0, tag=11)
            zeta = MPI.recv(1, tag=12)
            psiR = MPI.recv(2, tag=13)
        else:
            xi = MPI.recv(0, tag=11)
            zeta = MPI.recv(1, tag=12)
            psiR = MPI.recv(2, tag=13)
            psiT = MPI.recv(3, tag=14)

    elif MPI.size == 3:
        if MPI.rank == 0:
            xi = mimic.theory.pk2xi(r, kh, Dz2*pk, kmin=sim_kmin, kmax=sim_kmax,
                                    kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6), Rg=Rg)
            MPI.send(xi, to_rank=None, tag=11)
            zeta = mimic.theory.pk2zeta(r, kh, Dz2*pk, fk=fz0, kmin=sim_kmin, kmax=sim_kmax,
                                        kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6), Rg=Rg,
                                        cons_type=params["Constraints"]["Type"])
            MPI.send(zeta, to_rank=None, tag=12)
            psiR = MPI.recv(2, tag=13)
            psiT = MPI.recv(3, tag=14)
        elif MPI.rank == 1:
            psiR = mimic.theory.pk2psiR(r, kh, Dz2*pk, fk=fz0, kmin=sim_kmin, kmax=sim_kmax,
                                        kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6),
                                        Rg=Rg, cons_type=params["Constraints"]["Type"])
            MPI.send(psiR, to_rank=None, tag=13)
            xi = MPI.recv(0, tag=11)
            zeta = MPI.recv(0, tag=12)
            psiT = MPI.recv(2, tag=14)
        elif MPI.rank == 2:
            psiT = mimic.theory.pk2psiT(r, kh, Dz2*pk, fk=fz0, kmin=sim_kmin, kmax=sim_kmax,
                                        kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6),
                                        Rg=Rg, cons_type=params["Constraints"]["Type"])
            MPI.send(psiT, to_rank=None, tag=14)
            xi = MPI.recv(0, tag=11)
            zeta = MPI.recv(0, tag=12)
            psiR = MPI.recv(2, tag=13)

    elif MPI.size == 2:
        if MPI.rank == 0:
            xi = mimic.theory.pk2xi(r, kh, Dz2*pk, kmin=sim_kmin, kmax=sim_kmax,
                                    kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6), Rg=Rg)
            MPI.send(xi, to_rank=None, tag=11)
            zeta = mimic.theory.pk2zeta(r, kh, Dz2*pk, fk=fz0, kmin=sim_kmin, kmax=sim_kmax,
                                        kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6),
                                        Rg=Rg, cons_type=params["Constraints"]["Type"])
            MPI.send(zeta, to_rank=None, tag=12)
            psiR = MPI.recv(1, tag=13)
            psiT = MPI.recv(1, tag=14)
        elif MPI.rank == 1:
            psiR = mimic.theory.pk2psiR(r, kh, Dz2*pk, fk=fz0, kmin=sim_kmin, kmax=sim_kmax,
                                        kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6),
                                        Rg=Rg, cons_type=params["Constraints"]["Type"])
            MPI.send(psiR, to_rank=None, tag=13)
            psiT = mimic.theory.pk2psiT(r, kh, Dz2*pk, fk=fz0, kmin=sim_kmin, kmax=sim_kmax,
                                        kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6),
                                        Rg=Rg, cons_type=params["Constraints"]["Type"])
            MPI.send(psiT, to_rank=None, tag=14)
            xi = MPI.recv(0, tag=11)
            zeta = MPI.recv(0, tag=12)

    else:
        xi = mimic.theory.pk2xi(r, kh, Dz2*pk, kmin=sim_kmin, kmax=sim_kmax, kfactor=100,
                                kbinsmin=int(1e4), kbinsmax=int(1e6), Rg=Rg)
        zeta = mimic.theory.pk2zeta(r, kh, Dz2*pk, fk=fz0, kmin=sim_kmin, kmax=sim_kmax,
                                    kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6),
                                    Rg=Rg, cons_type=params["Constraints"]["Type"])
        psiR = mimic.theory.pk2psiR(r, kh, Dz2*pk, fk=fz0, kmin=sim_kmin, kmax=sim_kmax,
                                    kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6),
                                    Rg=Rg, cons_type=params["Constraints"]["Type"])
        psiT = mimic.theory.pk2psiT(r, kh, Dz2*pk, fk=fz0, kmin=sim_kmin, kmax=sim_kmax,
                                    kfactor=100, kbinsmin=int(1e4), kbinsmax=int(1e6),
                                    Rg=Rg, cons_type=params["Constraints"]["Type"])

    psiT0 = psiT[0]

    r = np.concatenate([np.array([0.]), r])
    xi = np.concatenate([np.array([xi[0]]), xi])
    psiR = np.concatenate([np.array([psiR[0]]), psiR])
    psiT = np.concatenate([np.array([psiT[0]]), psiT])
    zeta = np.concatenate([np.array([0.]), zeta])

    interp_xi   = interp1d(r, xi, kind='cubic', bounds_error=False, fill_value=0.)
    interp_zeta = interp1d(r, zeta, kind='cubic', bounds_error=False, fill_value=0.)
    interp_psiR = interp1d(r, psiR, kind='cubic', bounds_error=False, fill_value=0.)
    interp_psiT = interp1d(r, psiT, kind='cubic', bounds_error=False, fill_value=0.)

    if MPI.rank == 0:
        fname = params['Siminfo']['Simname'] + '_correlation_functions.npz'
        np.savez(fname, r=r, xi=xi, psiR=psiR, psiT=psiT, zeta=zeta)

    if MPI.rank == 0:
        printspace(verbose)
        printout("Compute covariance", 1, verbose)

    x1, x2 = MPI.create_split_ndgrid([cons["x"], cons["x"]], [False, True])
    y1, y2 = MPI.create_split_ndgrid([cons["y"], cons["y"]], [False, True])
    z1, z2 = MPI.create_split_ndgrid([cons["z"], cons["z"]], [False, True])

    ex1, ex2 = MPI.create_split_ndgrid([cons["ex"], cons["ex"]], [False, True])
    ey1, ey2 = MPI.create_split_ndgrid([cons["ey"], cons["ey"]], [False, True])
    ez1, ez2 = MPI.create_split_ndgrid([cons["ez"], cons["ez"]], [False, True])

    cov_uu = mimic.theory.get_cov_uu(x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2,
                                     interp_psiR, interp_psiT, z0,
                                     interp_Hz, psiT0, cons_type=params["Constraints"]["Type"])

    if MPI.rank == 0:
        covs_uu = [cov_uu]
        for i in range(1, MPI.size):
            covs_uu.append(MPI.recv(i, tag=10+i))
    else:
        MPI.send(cov_uu, 0, tag=10+MPI.rank)

    MPI.wait()

    if MPI.rank == 0:
        cov_uu = np.concatenate(covs_uu)
        cov_uu += np.diag(cons["u_err"]**2.)# + np.diag(np.ones(len(cons["u_err"])))*params["Constraints"]["sigma_NR"]**2.

        printout("Inverting covariance", 1, verbose)
        inv_uu = np.linalg.inv(cov_uu)

        printout("Compute eta vector", 1, verbose)
        eta = inv_uu.dot(cons['u'])

        MPI.send(eta, tag=11)

    else:
        eta = MPI.recv(0, tag=11)

    MPI.wait()

if ERROR is False:
    if params["Run"]["WF_Den"] is True:

        if MPI.rank == 0:
            printspace(verbose)
            printout("> COMPUTE WIENER FILTERED DENSITY ---------------------------------<", 0, verbose)
            printspace(verbose)

        ngrid = params["Siminfo"]["Ngrid"]
        boxsize = params["Siminfo"]["Boxsize"]

        Ngrids = np.array([ngrid, ngrid, ngrid], dtype=int)

        xedges, x1d = shift.cart.grid1D(boxsize, ngrid)
        y1d = np.copy(x1d)
        z1d = np.copy(x1d)

        FFT = MPI.mpi_fft_start(Ngrids)

        A = MPI.mpi_fft_array(FFT)
        partition = MPI.check_partition(Ngrids, np.array(np.shape(A)))
        del A

        if MPI.rank == 0:
            printout("Create 3D grid", 1, verbose)
        x3d, y3d, z3d = MPI.create_split_ndgrid([x1d, y1d, z1d], partition)

        x_shape = np.shape(x3d)

        x3d = x3d.flatten()
        y3d = y3d.flatten()
        z3d = z3d.flatten()

        z0 = params["Constraints"]["z_eff"]
        Hz = interp_Hz(z0)
        if params["Constraints"]["Type"] == "Vel":
            adot = mimic.theory.z2a(z0)*Hz
        elif params["Constraints"]["Type"] == "Psi":
            adot = 1.

        def get_WF_dens(x1, y1, z1, cons, interp_zeta, adot, eta, MPI, i, total):
            rx = cons['x']-x1
            ry = cons['y']-y1
            rz = cons['z']-z1
            r = np.sqrt(rx**2. + ry**2. + rz**2.)
            norm_rx = np.copy(rx)/r
            norm_ry = np.copy(ry)/r
            norm_rz = np.copy(rz)/r
            cov_du = interp_zeta(r)
            du = - adot*cov_du*norm_rx*cons['ex'] - adot*cov_du*norm_ry*cons['ey'] - adot*cov_du*norm_rz*cons['ez']
            if MPI.rank == 0:
                mimic.utils.progress_bar(i, total, explanation="--> Computing Wiener Filtered Density", indexing=True)
            return du.dot(eta)

        if MPI.rank == 0:
            printout("Get WF density", 1, verbose)
        WF_dens = np.array([get_WF_dens(x3d[i], y3d[i], z3d[i], cons, interp_zeta, adot, eta, MPI, i, len(x3d)) for i in range(0, len(x3d))])

        x3d = x3d.reshape(x_shape)
        y3d = y3d.reshape(x_shape)
        z3d = z3d.reshape(x_shape)
        WF_dens = WF_dens.reshape(x_shape)

        fname = params['Siminfo']['Simname'] + '_WF_dens_'+str(MPI.rank)+'.npz'
        np.savez(fname, x3d=x3d, y3d=y3d, z3d=z3d, WF_dens=WF_dens)

# Stopped here
if ERROR is False:
    if params["Run"]["WF_Psi"] is True or params["Run"]["WF_Vel"] is True:
        if MPI.rank == 0:
            printspace(verbose)
            printout("> COMPUTE WIENER FILTERED PSI --------------------------------------<", 0, verbose)

        ngrid = params["Siminfo"]["Ngrid"]
        boxsize = params["Siminfo"]["Boxsize"]
        Ngrid = [ngrid, ngrid, ngrid]

        Ngrids = np.array([ngrid, ngrid, ngrid], dtype=int)

        if MPI.rank == 0:
            printspace(verbose)
            printout("Prepare FFT", 1, verbose)

        FFT = MPI.mpi_fft_start(Ngrids)

        kx = shift.cart.kgrid1D(boxsize, ngrid)
        ky = np.copy(kx)
        kz = np.copy(kx)

        if MPI.rank == 0:
            printout("Forward FFT", 1, verbose)

        WF_dens_k = shift.cart.mpi_fft3D(WF_dens, x_shape, boxsize, ngrid, FFT)

        partitionk = MPI.check_partition(Ngrids, np.array(np.shape(WF_dens_k)))
        kx3d, ky3d, kz3d = MPI.create_split_ndgrid([kx, ky, kz], partitionk)

        kmag = np.sqrt(kx3d**2. + ky3d**2. + kz3d**2.)

        if MPI.rank == 0:
            printout("Compute displacement field if Fourier space of WF density", 1, verbose)

        if params["Cosmology"]["ScaleDepGrowth"] is False or params["Run"]["WF_Psi"] is True:
            WF_psi_kx = np.zeros(np.shape(kx3d)) + np.zeros(np.shape(kx3d))*1j
            WF_psi_ky = np.zeros(np.shape(ky3d)) + np.zeros(np.shape(ky3d))*1j
            WF_psi_kz = np.zeros(np.shape(kz3d)) + np.zeros(np.shape(kz3d))*1j
            cond = np.where(kmag != 0.)
            WF_psi_kx[cond] = WF_dens_k[cond]*1j*kx3d[cond]/(kmag[cond]**2.)
            WF_psi_ky[cond] = WF_dens_k[cond]*1j*ky3d[cond]/(kmag[cond]**2.)
            WF_psi_kz[cond] = WF_dens_k[cond]*1j*kz3d[cond]/(kmag[cond]**2.)
            #
            # cond = np.where(kmag == 0.)
            # WF_psi_kx[cond[0], cond[1], cond[2]] = 0 + 0*1j
            # WF_psi_ky[cond[0], cond[1], cond[2]] = 0 + 0*1j
            # WF_psi_kz[cond[0], cond[1], cond[2]] = 0 + 0*1j

            if MPI.rank == 0:
                printout("Backward FFT", 1, verbose)

            _WF_psi_x = shift.cart.mpi_ifft3D(WF_psi_kx, x_shape, boxsize, ngrid, FFT)
            _WF_psi_y = shift.cart.mpi_ifft3D(WF_psi_ky, x_shape, boxsize, ngrid, FFT)
            _WF_psi_z = shift.cart.mpi_ifft3D(WF_psi_kz, x_shape, boxsize, ngrid, FFT)
            WF_psi_x = np.copy(_WF_psi_x.real)
            WF_psi_y = np.copy(_WF_psi_y.real)
            WF_psi_z = np.copy(_WF_psi_z.real)
            del _WF_psi_x
            del _WF_psi_y
            del _WF_psi_z

            fname = params['Siminfo']['Simname'] + '_WF_psi_'+str(MPI.rank)+'.npz'
            np.savez(fname, x3d=x3d, y3d=y3d, z3d=z3d, WF_psi_x=WF_psi_x,
                     WF_psi_y=WF_psi_y, WF_psi_z=WF_psi_z)

if ERROR is False:
    if params["Run"]["WF_Vel"] is True:
        if MPI.rank == 0:
            printspace(verbose)
            printout("> COMPUTE WIENER FILTERED VELOCITY ---------------------------------<", 0, verbose)

        if MPI.rank == 0:
            printspace(verbose)
            printout("Compute velocity field from displacement", 1, verbose)

        if params["Cosmology"]["ScaleDepGrowth"] is False:
            if MPI.rank == 0:
                printout("Scale-independent growth ==> multiply by adot*f", 1, verbose)

            z0 = params["Constraints"]["z_eff"]
            Hz = interp_Hz(z0)
            adot = mimic.theory.z2a(z0)*Hz
            fz = interp_fz(z0)

            WF_vel_x = adot*fz*WF_psi_x
            WF_vel_y = adot*fz*WF_psi_y
            WF_vel_z = adot*fz*WF_psi_z

            fname = params['Siminfo']['Simname'] + '_WF_vel_'+str(MPI.rank)+'.npz'
            np.savez(fname, x3d=x3d, y3d=y3d, z3d=z3d, WF_vel_x=WF_vel_x,
                     WF_vel_y=WF_vel_y, WF_vel_z=WF_vel_z)
        else:
            if MPI.rank == 0:
                printout("Scale-dependent growth ==> multiply by adot*f(k) in Fourier space", 1, verbose)

            z0 = params["Constraints"]["z_eff"]
            Hz = interp_Hz(z0)
            adot = mimic.theory.z2a(z0)*Hz
            interp_fk = interp_fzk_2_fk(z0)

            WF_psi_kx = shift.cart.mpi_fft3D(WF_psi_x, x_shape, boxsize, ngrid, FFT)
            WF_psi_ky = shift.cart.mpi_fft3D(WF_psi_y, x_shape, boxsize, ngrid, FFT)
            WF_psi_kz = shift.cart.mpi_fft3D(WF_psi_z, x_shape, boxsize, ngrid, FFT)

            WF_vel_kx = np.zeros(np.shape(kx3d)) + np.zeros(np.shape(kx3d))*1j
            WF_vel_ky = np.zeros(np.shape(ky3d)) + np.zeros(np.shape(ky3d))*1j
            WF_vel_kz = np.zeros(np.shape(kz3d)) + np.zeros(np.shape(kz3d))*1j
            cond = np.where(kmag != 0.)
            WF_vel_kx[cond] = adot*interp_fk(kmag[cond])*WF_psi_kx[cond]
            WF_vel_ky[cond] = adot*interp_fk(kmag[cond])*WF_psi_ky[cond]
            WF_vel_kz[cond] = adot*interp_fk(kmag[cond])*WF_psi_kz[cond]

            # cond = np.where(kmag == 0.)
            # WF_vel_kx[cond[0], cond[1], cond[2]] = 0 + 0*1j
            # WF_vel_ky[cond[0], cond[1], cond[2]] = 0 + 0*1j
            # WF_vel_kz[cond[0], cond[1], cond[2]] = 0 + 0*1j

            WF_vel_x = shift.cart.mpi_ifft3D(WF_vel_kx, x_shape, boxsize, ngrid, FFT)
            WF_vel_y = shift.cart.mpi_ifft3D(WF_vel_ky, x_shape, boxsize, ngrid, FFT)
            WF_vel_z = shift.cart.mpi_ifft3D(WF_vel_kz, x_shape, boxsize, ngrid, FFT)

            fname = params['Siminfo']['Simname'] + '_WF_vel_'+str(MPI.rank)+'.npz'
            np.savez(fname, x3d=x3d, y3d=y3d, z3d=z3d, WF_vel_x=WF_vel_x,
                     WF_vel_y=WF_vel_y, WF_vel_z=WF_vel_z)

if ERROR is False:
    if params["Run"]["RZA"] is True:
        if MPI.rank == 0:
            printspace(verbose)
            printout("> APPLY REVERSE ZEL'DOVICH APPROXIMATION ---------------------------<", 0, verbose)
            printspace(verbose)

            printout("Add buffer region for interpolation", 1, verbose)

        interp_WF_psi_x = []
        interp_WF_psi_y = []
        interp_WF_psi_z = []

        if MPI.rank == 0:
            fname = params['Siminfo']['Simname'] + '_WF_psi_'+str(MPI.size-1)+'.npz'
        else:
            fname = params['Siminfo']['Simname'] + '_WF_psi_'+str(MPI.rank-1)+'.npz'
        data = np.load(fname)

        interp_WF_psi_x.append(data['WF_psi_x'][-1])
        interp_WF_psi_y.append(data['WF_psi_y'][-1])
        interp_WF_psi_z.append(data['WF_psi_z'][-1])

        for i in range(0, len(data['WF_psi_x'])):
            interp_WF_psi_x.append(data['WF_psi_x'][i])
            interp_WF_psi_y.append(data['WF_psi_y'][i])
            interp_WF_psi_z.append(data['WF_psi_z'][i])

        if MPI.rank == MPI.size-1:
            fname = params['Siminfo']['Simname'] + '_WF_psi_'+str(0)+'.npz'
        else:
            fname = params['Siminfo']['Simname'] + '_WF_psi_'+str(MPI.rank+1)+'.npz'
        data = np.load(fname)

        interp_WF_psi_x.append(data['WF_psi_x'][0])
        interp_WF_psi_y.append(data['WF_psi_y'][0])
        interp_WF_psi_z.append(data['WF_psi_z'][0])

        if MPI.rank == 0:
            printout("Interpolate Psi at constraint points", 1, verbose)

        interp_WF_psi_x = np.array(interp_WF_psi_x, dtype=object)
        interp_WF_psi_y = np.array(interp_WF_psi_y, dtype=object)
        interp_WF_psi_z = np.array(interp_WF_psi_z, dtype=object)

        dx = boxsize/ngrid
        xmin = np.min(x3d) - dx/2. - dx
        xmax = np.max(x3d) + dx/2. + dx

        cond = np.where((cons['x'] >= xmin + dx) & (cons['x'] < xmax - dx))[0]
        if len(cond) > 0:
            RZA_psi_x = fiesta.interp.trilinear(interp_WF_psi_x, [xmax-xmin, boxsize, boxsize],
                                                cons['x'][cond], cons['y'][cond], cons['z'][cond], origin=[xmin, 0., 0.],
                                                fill_value=0., periodic=[False, True, True])
            RZA_psi_y = fiesta.interp.trilinear(interp_WF_psi_y, [xmax-xmin, boxsize, boxsize],
                                                cons['x'][cond], cons['y'][cond], cons['z'][cond], origin=[xmin, 0., 0.],
                                                fill_value=0., periodic=[False, True, True])
            RZA_psi_z = fiesta.interp.trilinear(interp_WF_psi_z, [xmax-xmin, boxsize, boxsize],
                                                cons['x'][cond], cons['y'][cond], cons['z'][cond], origin=[xmin, 0., 0.],
                                                fill_value=0., periodic=[False, True, True])

        # interp_WF_vel_x = []
        # interp_WF_vel_y = []
        # interp_WF_vel_z = []
        #
        # if MPI.rank == 0:
        #     fname = params['Siminfo']['Simname'] + '_WF_vel_'+str(MPI.size-1)+'.npz'
        # else:
        #     fname = params['Siminfo']['Simname'] + '_WF_vel_'+str(MPI.rank-1)+'.npz'
        # data = np.load(fname)
        #
        # interp_WF_vel_x.append(data['WF_vel_x'][-1])
        # interp_WF_vel_y.append(data['WF_vel_y'][-1])
        # interp_WF_vel_z.append(data['WF_vel_z'][-1])
        #
        # for i in range(0, len(data['WF_vel_x'])):
        #     interp_WF_vel_x.append(data['WF_vel_x'][i])
        #     interp_WF_vel_y.append(data['WF_vel_y'][i])
        #     interp_WF_vel_z.append(data['WF_vel_z'][i])
        #
        # if MPI.rank == MPI.size-1:
        #     fname = params['Siminfo']['Simname'] + '_WF_vel_'+str(0)+'.npz'
        # else:
        #     fname = params['Siminfo']['Simname'] + '_WF_vel_'+str(MPI.rank+1)+'.npz'
        # data = np.load(fname)
        #
        # interp_WF_vel_x.append(data['WF_vel_x'][0])
        # interp_WF_vel_y.append(data['WF_vel_y'][0])
        # interp_WF_vel_z.append(data['WF_vel_z'][0])
        #
        # if MPI.rank == 0:
        #     printout("Interpolate Vel at constraint points", 1, verbose)
        #
        # interp_WF_vel_x = np.array(interp_WF_vel_x, dtype=object)
        # interp_WF_vel_y = np.array(interp_WF_vel_y, dtype=object)
        # interp_WF_vel_z = np.array(interp_WF_vel_z, dtype=object)
        #
        # dx = boxsize/ngrid
        # xmin = np.min(x3d) - dx/2. - dx
        # xmax = np.max(x3d) + dx/2. + dx
        #
        # cond = np.where((cons['x'] >= xmin + dx) & (cons['x'] < xmax - dx))[0]
        # if len(cond) > 0:
        #     RZA_vel_x = fiesta.interp.trilinear(interp_WF_vel_x, [xmax-xmin, boxsize, boxsize],
        #                                         cons['x'][cond], cons['y'][cond], cons['z'][cond], origin=[xmin, 0., 0.],
        #                                         fill_value=0., periodic=[False, True, True])
        #     RZA_vel_y = fiesta.interp.trilinear(interp_WF_vel_y, [xmax-xmin, boxsize, boxsize],
        #                                         cons['x'][cond], cons['y'][cond], cons['z'][cond], origin=[xmin, 0., 0.],
        #                                         fill_value=0., periodic=[False, True, True])
        #     RZA_vel_z = fiesta.interp.trilinear(interp_WF_vel_z, [xmax-xmin, boxsize, boxsize],
        #                                         cons['x'][cond], cons['y'][cond], cons['z'][cond], origin=[xmin, 0., 0.],
        #                                         fill_value=0., periodic=[False, True, True])

        if MPI.rank == 0:
            printout("Perform Reverse Zel'dovich Approximation", 1, verbose)

        if MPI.rank != 0:
            if len(cond) > 0:
                MPI.send(cond, 0, tag=11)
                MPI.send(RZA_psi_x, 0, tag=12)
                MPI.send(RZA_psi_y, 0, tag=13)
                MPI.send(RZA_psi_z, 0, tag=14)
            else:
                MPI.send(None, 0, tag=11)
                MPI.send(None, 0, tag=12)
                MPI.send(None, 0, tag=13)
                MPI.send(None, 0, tag=14)
        else:
            cons['x_rza'] = np.copy(cons['x'])
            cons['y_rza'] = np.copy(cons['y'])
            cons['z_rza'] = np.copy(cons['z'])
            cons['x_rza'][cond] -= RZA_psi_x
            cons['y_rza'][cond] -= RZA_psi_y
            cons['z_rza'][cond] -= RZA_psi_z
            cons['u_rza'] = np.copy(cons['u'])
            cons['u_err_rza'] = np.zeros(len(cons['u_err']))
            cons['ex_rza'] = np.copy(cons['ex'])
            cons['ey_rza'] = np.copy(cons['ey'])
            cons['ez_rza'] = np.copy(cons['ez'])
            cons['u_rza'][cond] = np.sqrt(RZA_psi_x**2. + RZA_psi_y**2. + RZA_psi_z**2.)
            cons['ex_rza'][cond] = RZA_psi_x/cons['u_rza'][cond]
            cons['ey_rza'][cond] = RZA_psi_y/cons['u_rza'][cond]
            cons['ez_rza'][cond] = RZA_psi_z/cons['u_rza'][cond]
            for i in range(1, MPI.size):
                cond = MPI.recv(i, tag=11)
                RZA_psi_x = MPI.recv(i, tag=12)
                RZA_psi_y = MPI.recv(i, tag=13)
                RZA_psi_z = MPI.recv(i, tag=14)
                if cond is not None:
                    cons['x_rza'][cond] -= RZA_psi_x
                    cons['y_rza'][cond] -= RZA_psi_y
                    cons['z_rza'][cond] -= RZA_psi_z
                    cons['u_rza'][cond] = np.sqrt(RZA_psi_x**2. + RZA_psi_y**2. + RZA_psi_z**2.)
                    cons['ex_rza'][cond] = RZA_psi_x/cons['u_rza'][cond]
                    cons['ey_rza'][cond] = RZA_psi_y/cons['u_rza'][cond]
                    cons['ez_rza'][cond] = RZA_psi_z/cons['u_rza'][cond]

        MPI.wait()
        #
        # cond = np.where((cons['x'] >= xmin + dx) & (cons['x'] < xmax - dx))[0]
        #
        # if MPI.rank != 0:
        #     if len(cond) > 0:
        #         MPI.send(cond, 0, tag=11)
        #         MPI.send(RZA_vel_x, 0, tag=12)
        #         MPI.send(RZA_vel_y, 0, tag=13)
        #         MPI.send(RZA_vel_z, 0, tag=14)
        #     else:
        #         MPI.send(None, 0, tag=11)
        #         MPI.send(None, 0, tag=12)
        #         MPI.send(None, 0, tag=13)
        #         MPI.send(None, 0, tag=14)
        # else:
        #     cons['u_rza'] = np.copy(cons['u'])
        #     cons['u_err_rza'] = np.zeros(len(cons['u_err']))
        #     cons['ex_rza'] = np.copy(cons['ex'])
        #     cons['ey_rza'] = np.copy(cons['ey'])
        #     cons['ez_rza'] = np.copy(cons['ez'])
        #     cons['u_rza'][cond] = np.sqrt(RZA_vel_x**2. + RZA_vel_y**2. + RZA_vel_z**2.)
        #     cons['ex_rza'][cond] = RZA_vel_x/cons['u_rza'][cond]
        #     cons['ey_rza'][cond] = RZA_vel_y/cons['u_rza'][cond]
        #     cons['ez_rza'][cond] = RZA_vel_z/cons['u_rza'][cond]
        #     for i in range(1, MPI.size):
        #         cond = MPI.recv(i, tag=11)
        #         RZA_vel_x = MPI.recv(i, tag=12)
        #         RZA_vel_y = MPI.recv(i, tag=13)
        #         RZA_vel_z = MPI.recv(i, tag=14)
        #         if cond is not None:
        #             cons['u_rza'][cond] = np.sqrt(RZA_vel_x**2. + RZA_vel_y**2. + RZA_vel_z**2.)
        #             cons['ex_rza'][cond] = RZA_vel_x/cons['u_rza'][cond]
        #             cons['ey_rza'][cond] = RZA_vel_y/cons['u_rza'][cond]
        #             cons['ez_rza'][cond] = RZA_vel_z/cons['u_rza'][cond]
        # MPI.wait()

        # if params["Cosmology"]["ScaleDepGrowth"] is False:
        #     if MPI.rank == 0:
        #         printout("Scale-independent growth ==> multiply by adot*f", 1, verbose)
        #
        #     z0 = params["Constraints"]["z_eff"]
        #     Hz = interp_Hz(z0)
        #     adot = mimic.theory.z2a(z0)*Hz
        #     fz = interp_fz(z0)
        #
        #     vel_x = adot*fz*psi_x
        #     vel_y = adot*fz*psi_y
        #     vel_z = adot*fz*psi_z
        #
        #     fname = params['Siminfo']['Simname'] + '_vel_rand_'+str(MPI.rank)+'.npz'
        #     np.savez(fname, x3d=x3d, y3d=y3d, z3d=z3d, vel_x=vel_x, vel_y=vel_y, vel_z=vel_z)
        #
        # else:
        #     if MPI.rank == 0:
        #         printout("Scale-dependent growth ==> multiply by adot*f(k) in Fourier space", 1, verbose)
        #
        #     z0 = params["Constraints"]["z_eff"]
        #     Hz = interp_Hz(z0)
        #     adot = mimic.theory.z2a(z0)*Hz
        #     interp_fk = interp_fzk_2_fk(z0)
        #
        #     psi_kx = shift.cart.forward_mpi_fft_3D(psi_x, x_shape, boxsize, ngrid, FFT)
        #     psi_ky = shift.cart.forward_mpi_fft_3D(psi_y, x_shape, boxsize, ngrid, FFT)
        #     psi_kz = shift.cart.forward_mpi_fft_3D(psi_z, x_shape, boxsize, ngrid, FFT)
        #
        #     vel_kx = adot*interp_fk(kmag)*psi_kx
        #     vel_ky = adot*interp_fk(kmag)*psi_ky
        #     vel_kz = adot*interp_fk(kmag)*psi_kz
        #
        #     cond = np.where(kmag == 0.)
        #     vel_kx[cond] = 0 + 0*1j
        #     vel_ky[cond] = 0 + 0*1j
        #     vel_kz[cond] = 0 + 0*1j
        #
        #     vel_x = shift.cart.backward_mpi_fft_3D(vel_kx, x_shape, boxsize, ngrid, FFT)
        #     vel_y = shift.cart.backward_mpi_fft_3D(vel_ky, x_shape, boxsize, ngrid, FFT)
        #     vel_z = shift.cart.backward_mpi_fft_3D(vel_kz, x_shape, boxsize, ngrid, FFT)
        #
        #     fname = params['Siminfo']['Simname'] + '_vel_rand_'+str(MPI.rank)+'.npz'
        #     np.savez(fname, x3d=x3d, y3d=y3d, z3d=z3d, vel_x=vel_x, vel_y=vel_y, vel_z=vel_z)

        MPI.wait()

        if MPI.rank == 0:
            fname = params['Siminfo']['Simname'] + '_cons_RZA.npz'
            np.savez(fname, x=cons['x_rza']-halfsize, y=cons['y_rza']-halfsize,
                     z=cons['z_rza']-halfsize, ex=cons['ex_rza'], ey=cons['ey_rza'], ez=cons['ez_rza'],
                     u=cons['u_rza'], u_err=cons['u_err_rza'])


if ERROR is False:
    if params["Run"]["CR"] is True or params["Run"]["IC"] is True:
        if MPI.rank == 0:
            printspace(verbose)
            printout("> CONSTRUCT RANDOM REALISATION -------------------------------------<", 0, verbose)

        if MPI.rank == 0:
            printspace(verbose)
            printout("> Construct random realisation", 0, verbose)
            printout("Seed = "+str(params["Randoms"]["Seed"]), 1, verbose)

        seed = params["Randoms"]["Seed"] + MPI.rank

        ngrid = params["Siminfo"]["Ngrid"]
        boxsize = params["Siminfo"]["Boxsize"]
        dx = boxsize/ngrid

        Ngrids = np.array([ngrid, ngrid, ngrid], dtype=int)

        xedges, x1d = shift.cart.grid1D(boxsize, ngrid)
        y1d = np.copy(x1d)
        z1d = np.copy(x1d)

        FFT = MPI.mpi_fft_start(Ngrids)

        A = MPI.mpi_fft_array(FFT)
        partition = MPI.check_partition(Ngrids, np.array(np.shape(A)))
        del A

        if MPI.rank == 0:
            printout("Create 3D grid", 1, verbose)
        x3d, y3d, z3d = MPI.create_split_ndgrid([x1d, y1d, z1d], partition)

        x_shape = np.shape(x3d)

        if MPI.rank == 0:
            printout("Create white noise field", 1, verbose)

        WN = mimic.randoms.get_white_noise(seed, *x_shape)

        kx = shift.cart.kgrid1D(boxsize, ngrid)
        ky = np.copy(kx)
        kz = np.copy(kx)

        if MPI.rank == 0:
            printout("Forward FFT of white noise", 1, verbose)

        WN_k = shift.cart.mpi_fft3D(WN, x_shape, boxsize, ngrid, FFT)

        partitionk = MPI.check_partition(Ngrids, np.array(np.shape(WN_k)))
        kx3d, ky3d, kz3d = MPI.create_split_ndgrid([kx, ky, kz], partitionk)

        kmag = np.sqrt(kx3d**2. + ky3d**2. + kz3d**2.)

        if MPI.rank == 0:
            printout("Color white noise", 1, verbose)

        dens_k = mimic.randoms.color_white_noise(WN_k, dx, kmag, interp_pk)

        if MPI.rank == 0:
            printout("Backward FFT of colored noise", 1, verbose)

        _dens = shift.cart.mpi_ifft3D(dens_k, x_shape, boxsize, ngrid, FFT)
        dens = np.copy(_dens.real)
        del _dens

        fname = params['Siminfo']['Simname'] + '_dens_rand_'+str(MPI.rank)+'.npz'
        np.savez(fname, x3d=x3d, y3d=y3d, z3d=z3d, dens=dens)

        if MPI.rank == 0:
            printout("Compute Psi", 1, verbose)

        psi_kx = np.zeros(np.shape(kx3d)) + np.zeros(np.shape(kx3d))*1j
        psi_ky = np.zeros(np.shape(ky3d)) + np.zeros(np.shape(ky3d))*1j
        psi_kz = np.zeros(np.shape(kz3d)) + np.zeros(np.shape(kz3d))*1j
        cond = np.where(kmag != 0.)
        psi_kx[cond] = dens_k[cond]*1j*kx3d[cond]/(kmag[cond]**2.)
        psi_ky[cond] = dens_k[cond]*1j*ky3d[cond]/(kmag[cond]**2.)
        psi_kz[cond] = dens_k[cond]*1j*kz3d[cond]/(kmag[cond]**2.)

        # psi_kx = dens_k*1j*kx3d/(kmag**2.)
        # psi_ky = dens_k*1j*ky3d/(kmag**2.)
        # psi_kz = dens_k*1j*kz3d/(kmag**2.)

        # cond = np.where(kmag == 0.)
        # psi_kx[cond] = 0 + 0*1j
        # psi_ky[cond] = 0 + 0*1j
        # psi_kz[cond] = 0 + 0*1j

        if MPI.rank == 0:
            printout("Backward FFT of Psi", 1, verbose)

        _psi_x = shift.cart.mpi_ifft3D(psi_kx, x_shape, boxsize, ngrid, FFT)
        _psi_y = shift.cart.mpi_ifft3D(psi_ky, x_shape, boxsize, ngrid, FFT)
        _psi_z = shift.cart.mpi_ifft3D(psi_kz, x_shape, boxsize, ngrid, FFT)
        psi_x = np.copy(_psi_x.real)
        psi_y = np.copy(_psi_y.real)
        psi_z = np.copy(_psi_z.real)
        del _psi_x
        del _psi_y
        del _psi_z

        fname = params['Siminfo']['Simname'] + '_psi_rand_'+str(MPI.rank)+'.npz'
        np.savez(fname, x3d=x3d, y3d=y3d, z3d=z3d, psi_x=psi_x, psi_y=psi_y, psi_z=psi_z)

        if params["Cosmology"]["ScaleDepGrowth"] is False:
            if MPI.rank == 0:
                printout("Scale-independent growth ==> multiply by adot*f", 1, verbose)

            z0 = params["Constraints"]["z_eff"]
            Hz = interp_Hz(z0)
            adot = mimic.theory.z2a(z0)*Hz
            fz = interp_fz(z0)

            vel_x = adot*fz*psi_x
            vel_y = adot*fz*psi_y
            vel_z = adot*fz*psi_z

            fname = params['Siminfo']['Simname'] + '_vel_rand_'+str(MPI.rank)+'.npz'
            np.savez(fname, x3d=x3d, y3d=y3d, z3d=z3d, vel_x=vel_x, vel_y=vel_y, vel_z=vel_z)

        else:
            if MPI.rank == 0:
                printout("Scale-dependent growth ==> multiply by adot*f(k) in Fourier space", 1, verbose)

            z0 = params["Constraints"]["z_eff"]
            Hz = interp_Hz(z0)
            adot = mimic.theory.z2a(z0)*Hz
            interp_fk = interp_fzk_2_fk(z0)

            psi_kx = shift.cart.mpi_fft3D(psi_x, x_shape, boxsize, ngrid, FFT)
            psi_ky = shift.cart.mpi_fft3D(psi_y, x_shape, boxsize, ngrid, FFT)
            psi_kz = shift.cart.mpi_fft3D(psi_z, x_shape, boxsize, ngrid, FFT)

            vel_kx = np.zeros(np.shape(kx3d)) + np.zeros(np.shape(kx3d))*1j
            vel_ky = np.zeros(np.shape(ky3d)) + np.zeros(np.shape(ky3d))*1j
            vel_kz = np.zeros(np.shape(kz3d)) + np.zeros(np.shape(kz3d))*1j
            cond = np.where(kmag != 0.)
            vel_kx[cond] = adot*interp_fk(kmag[cond])*psi_kx[cond]
            vel_ky[cond] = adot*interp_fk(kmag[cond])*psi_ky[cond]
            vel_kz[cond] = adot*interp_fk(kmag[cond])*psi_kz[cond]

            # vel_kx = adot*interp_fk(kmag)*psi_kx
            # vel_ky = adot*interp_fk(kmag)*psi_ky
            # vel_kz = adot*interp_fk(kmag)*psi_kz
            #
            # cond = np.where(kmag == 0.)
            # vel_kx[cond[0], cond[1], cond[2]] = 0 + 0*1j
            # vel_ky[cond[0], cond[1], cond[2]] = 0 + 0*1j
            # vel_kz[cond[0], cond[1], cond[2]] = 0 + 0*1j

            vel_x = shift.cart.mpi_ifft3D(vel_kx, x_shape, boxsize, ngrid, FFT)
            vel_y = shift.cart.mpi_ifft3D(vel_ky, x_shape, boxsize, ngrid, FFT)
            vel_z = shift.cart.mpi_ifft3D(vel_kz, x_shape, boxsize, ngrid, FFT)

            fname = params['Siminfo']['Simname'] + '_vel_rand_'+str(MPI.rank)+'.npz'
            np.savez(fname, x3d=x3d, y3d=y3d, z3d=z3d, vel_x=vel_x, vel_y=vel_y, vel_z=vel_z)

        MPI.wait()

if ERROR is False:
    if params["Run"]["CR"] is True and params["Constraints"] is not None:

        if MPI.rank == 0:
            printspace(verbose)
            printout("> CONSTRUCT CONSTRAINED REALISATION --------------------------------<", 0, verbose)
            printspace(verbose)

        if params["Constraints"]["Type"] == "Vel":
            interp_vel_x = []
            interp_vel_y = []
            interp_vel_z = []
            if MPI.rank == 0:
                fname = params['Siminfo']['Simname'] + '_vel_rand_'+str(MPI.size-1)+'.npz'
            else:
                fname = params['Siminfo']['Simname'] + '_vel_rand_'+str(MPI.rank-1)+'.npz'
        elif params["Constraints"]["Type"] == "Psi":
            interp_vel_x = []
            interp_vel_y = []
            interp_vel_z = []
            if MPI.rank == 0:
                fname = params['Siminfo']['Simname'] + '_psi_rand_'+str(MPI.size-1)+'.npz'
            else:
                fname = params['Siminfo']['Simname'] + '_psi_rand_'+str(MPI.rank-1)+'.npz'

        data = np.load(fname)

        if params["Constraints"]["Type"] == "Vel":
            interp_vel_x.append(data['vel_x'][-1])
            interp_vel_y.append(data['vel_y'][-1])
            interp_vel_z.append(data['vel_z'][-1])
        elif params["Constraints"]["Type"] == "Psi":
            interp_vel_x.append(data['psi_x'][-1])
            interp_vel_y.append(data['psi_y'][-1])
            interp_vel_z.append(data['psi_z'][-1])

        if params["Constraints"]["Type"] == "Vel":
            for i in range(0, len(data['vel_x'])):
                interp_vel_x.append(data['vel_x'][i])
                interp_vel_y.append(data['vel_y'][i])
                interp_vel_z.append(data['vel_z'][i])
        elif params["Constraints"]["Type"] == "Psi":
            for i in range(0, len(data['psi_x'])):
                interp_vel_x.append(data['psi_x'][i])
                interp_vel_y.append(data['psi_y'][i])
                interp_vel_z.append(data['psi_z'][i])

        if params["Constraints"]["Type"] == "Vel":
            if MPI.rank == MPI.size-1:
                fname = params['Siminfo']['Simname'] + '_vel_rand_'+str(0)+'.npz'
            else:
                fname = params['Siminfo']['Simname'] + '_vel_rand_'+str(MPI.rank+1)+'.npz'
        elif params["Constraints"]["Type"] == "Psi":
            if MPI.rank == MPI.size-1:
                fname = params['Siminfo']['Simname'] + '_psi_rand_'+str(0)+'.npz'
            else:
                fname = params['Siminfo']['Simname'] + '_psi_rand_'+str(MPI.rank+1)+'.npz'

        data = np.load(fname)

        if params["Constraints"]["Type"] == "Vel":
            interp_vel_x.append(data['vel_x'][0])
            interp_vel_y.append(data['vel_y'][0])
            interp_vel_z.append(data['vel_z'][0])
        elif params["Constraints"]["Type"] == "Psi":
            interp_vel_x.append(data['psi_x'][0])
            interp_vel_y.append(data['psi_y'][0])
            interp_vel_z.append(data['psi_z'][0])

        if MPI.rank == 0:
            printout("Interpolate random velocities at constraint points", 1, verbose)

        if params["Constraints"]["Type"] == "Vel":
            interp_vel_x = np.array(interp_vel_x, dtype=object)
            interp_vel_y = np.array(interp_vel_y, dtype=object)
            interp_vel_z = np.array(interp_vel_z, dtype=object)
        elif params["Constraints"]["Type"] == "Psi":
            interp_vel_x = np.array(interp_vel_x, dtype=object)
            interp_vel_y = np.array(interp_vel_y, dtype=object)
            interp_vel_z = np.array(interp_vel_z, dtype=object)

        dx = boxsize/ngrid
        xmin = np.min(x3d) - dx/2. - dx
        xmax = np.max(x3d) + dx/2. + dx

        cond = np.where((cons['x'] >= xmin + dx) & (cons['x'] < xmax - dx))[0]
        if len(cond) > 0:

            cons_vel_x = fiesta.interp.trilinear(interp_vel_x, [xmax-xmin, boxsize, boxsize],
                                                 cons['x'][cond], cons['y'][cond], cons['z'][cond], origin=[xmin, 0., 0.],
                                                 fill_value=0., periodic=[False, True, True])
            cons_vel_y = fiesta.interp.trilinear(interp_vel_y, [xmax-xmin, boxsize, boxsize],
                                                 cons['x'][cond], cons['y'][cond], cons['z'][cond], origin=[xmin, 0., 0.],
                                                 fill_value=0., periodic=[False, True, True])
            cons_vel_z = fiesta.interp.trilinear(interp_vel_z, [xmax-xmin, boxsize, boxsize],
                                                 cons['x'][cond], cons['y'][cond], cons['z'][cond], origin=[xmin, 0., 0.],
                                                 fill_value=0., periodic=[False, True, True])

        if MPI.rank != 0:
            if len(cond) > 0:
                MPI.send(cond, 0, tag=11)
                MPI.send(cons_vel_x, 0, tag=12)
                MPI.send(cons_vel_y, 0, tag=13)
                MPI.send(cons_vel_z, 0, tag=14)
            else:
                MPI.send(None, 0, tag=11)
                MPI.send(None, 0, tag=12)
                MPI.send(None, 0, tag=13)
                MPI.send(None, 0, tag=14)
            eta = MPI.recv(0, tag=15)
        else:
            ux_rand = np.copy(cons['x'])
            uy_rand = np.copy(cons['x'])
            uz_rand = np.copy(cons['x'])
            if len(cond) > 0:
                ux_rand[cond] = cons_vel_x
                uy_rand[cond] = cons_vel_y
                uz_rand[cond] = cons_vel_z

            for i in range(1, MPI.size):
                cond = MPI.recv(i, tag=11)
                cons_vel_x = MPI.recv(i, tag=12)
                cons_vel_y = MPI.recv(i, tag=13)
                cons_vel_z = MPI.recv(i, tag=14)
                if cond is not None:
                    ux_rand[cond] = cons_vel_x
                    uy_rand[cond] = cons_vel_y
                    uz_rand[cond] = cons_vel_z

            u_rand = ux_rand*cons["ex"] + uy_rand*cons["ey"] + uz_rand*cons["ez"]

            printout("Compute eta vector", 1, verbose)
            eta = inv_uu.dot(cons['u']-u_rand)

            MPI.send(eta, tag=15)

        MPI.wait()

        x_shape = np.shape(x3d)

        x3d = x3d.flatten()
        y3d = y3d.flatten()
        z3d = z3d.flatten()
        dens = dens.flatten()

        z0 = params["Constraints"]["z_eff"]
        Hz = interp_Hz(z0)
        if params["Constraints"]["Type"] == "Vel":
            adot = mimic.theory.z2a(z0)*Hz
        elif params["Constraints"]["Type"] == "Psi":
            adot = 1.

        def get_CR_dens(x1, y1, z1, dens, cons, interp_zeta, adot, eta, MPI, i, total):
            rx = cons['x']-x1
            ry = cons['y']-y1
            rz = cons['z']-z1
            r = np.sqrt(rx**2. + ry**2. + rz**2.)
            norm_rx = np.copy(rx)/r
            norm_ry = np.copy(ry)/r
            norm_rz = np.copy(rz)/r
            cov_du = interp_zeta(r)
            du = - adot*cov_du*norm_rx*cons['ex'] - adot*cov_du*norm_ry*cons['ey'] - adot*cov_du*norm_rz*cons['ez']
            if MPI.rank == 0:
                mimic.utils.progress_bar(i, total, explanation="--> Computing Constrained Density", indexing=True)
            return dens + du.dot(eta)

        if MPI.rank == 0:
            printout("Get CR density", 1, verbose)

        dens_CR = np.array([get_CR_dens(x3d[i], y3d[i], z3d[i], dens[i], cons, interp_zeta, adot, eta, MPI, i, len(x3d)) for i in range(0, len(x3d))])

        x3d = x3d.reshape(x_shape)
        y3d = y3d.reshape(x_shape)
        z3d = z3d.reshape(x_shape)
        dens_CR = dens_CR.reshape(x_shape)

        fname = params['Siminfo']['Simname'] + '_dens_CR_'+str(MPI.rank)+'.npz'
        np.savez(fname, x3d=x3d, y3d=y3d, z3d=z3d, dens=dens_CR)


if ERROR is False:
    if params["Run"]["IC"] is True:

        if MPI.rank == 0:
            printspace(verbose)
            printout("> CONSTRUCT INITIAL CONDITIONS -------------------------------------<", 0, verbose)
            printspace(verbose)

        if params["Run"]["CR"] is True:
            dens = dens_CR

        if params["Constraints"] is None:
            z0 = 0.
        else:
            z0 = params["Constraints"]["z_eff"]

        z_init = params["ICs"]["z_init"]

        if MPI.rank == 0:
            printout("Multiply density by linear growth function for initial redshift", 1, verbose)


        if params["Cosmology"]["ScaleDepGrowth"] is False:
            dens *= interp_Dz(z_init)/interp_Dz(z0)
        else:
            interp_Dk0 = interp_Dzk_2_Dk(z0)
            interp_Dki = interp_Dzk_2_Dk(z_init)

            dens_k = shift.cart.mpi_fft3D(dens, x_shape, boxsize, ngrid, FFT)

            dens_k *= interp_Dki(kmag)/interp_Dk0(kmag)

            cond = np.where(kmag == 0.)
            dens_k[cond] = 0 + 0*1j

            dens = shift.cart.mpi_ifft3D(dens_k, x_shape, boxsize, ngrid, FFT)

        dens_k = shift.cart.mpi_fft3D(dens, x_shape, boxsize, ngrid, FFT)

        if MPI.rank == 0:
            printout("Compute Psi", 1, verbose)

        psi_kx = np.zeros(np.shape(kx3d)) + np.zeros(np.shape(kx3d))*1j
        psi_ky = np.zeros(np.shape(ky3d)) + np.zeros(np.shape(ky3d))*1j
        psi_kz = np.zeros(np.shape(kz3d)) + np.zeros(np.shape(kz3d))*1j
        cond = np.where(kmag != 0.)
        psi_kx[cond] = dens_k[cond]*1j*kx3d[cond]/(kmag[cond]**2.)
        psi_ky[cond] = dens_k[cond]*1j*ky3d[cond]/(kmag[cond]**2.)
        psi_kz[cond] = dens_k[cond]*1j*kz3d[cond]/(kmag[cond]**2.)

        # psi_kx = dens_k*1j*kx3d/(kmag**2.)
        # psi_ky = dens_k*1j*ky3d/(kmag**2.)
        # psi_kz = dens_k*1j*kz3d/(kmag**2.)
        #
        # cond = np.where(kmag == 0.)
        # psi_kx[cond[0], cond[1], cond[2]] = 0 + 0*1j
        # psi_ky[cond[0], cond[1], cond[2]] = 0 + 0*1j
        # psi_kz[cond[0], cond[1], cond[2]] = 0 + 0*1j

        if MPI.rank == 0:
            printout("Backward FFT of Psi", 1, verbose)

        _psi_x = shift.cart.mpi_ifft3D(psi_kx, x_shape, boxsize, ngrid, FFT)
        _psi_y = shift.cart.mpi_ifft3D(psi_ky, x_shape, boxsize, ngrid, FFT)
        _psi_z = shift.cart.mpi_ifft3D(psi_kz, x_shape, boxsize, ngrid, FFT)
        psi_x = np.copy(_psi_x.real)
        psi_y = np.copy(_psi_y.real)
        psi_z = np.copy(_psi_z.real)
        del _psi_x
        del _psi_y
        del _psi_z

        pos_x = x3d + psi_x
        pos_y = y3d + psi_y
        pos_z = z3d + psi_z

        if params["Cosmology"]["ScaleDepGrowth"] is False:
            if MPI.rank == 0:
                printout("Scale-independent growth ==> multiply by adot*f", 1, verbose)

            Hz = interp_Hz(z_init)
            adot = mimic.theory.z2a(z_init)*Hz
            fz = interp_fz(z_init)

            vel_x = adot*fz*psi_x
            vel_y = adot*fz*psi_y
            vel_z = adot*fz*psi_z

        else:
            if MPI.rank == 0:
                printout("Scale-dependent growth ==> multiply by adot*f(k) in Fourier space", 1, verbose)

            Hz = interp_Hz(z_init)
            adot = mimic.theory.z2a(z_init)*Hz
            interp_fk = interp_fzk_2_fk(z_init)

            psi_kx = shift.cart.mpi_fft3D(psi_x, x_shape, boxsize, ngrid, FFT)
            psi_ky = shift.cart.mpi_fft3D(psi_y, x_shape, boxsize, ngrid, FFT)
            psi_kz = shift.cart.mpi_fft3D(psi_z, x_shape, boxsize, ngrid, FFT)

            vel_kx = np.zeros(np.shape(kx3d)) + np.zeros(np.shape(kx3d))*1j
            vel_ky = np.zeros(np.shape(ky3d)) + np.zeros(np.shape(ky3d))*1j
            vel_kz = np.zeros(np.shape(kz3d)) + np.zeros(np.shape(kz3d))*1j
            cond = np.where(kmag != 0.)
            vel_kx[cond] = adot*interp_fk(kmag[cond])*psi_kx[cond]
            vel_ky[cond] = adot*interp_fk(kmag[cond])*psi_ky[cond]
            vel_kz[cond] = adot*interp_fk(kmag[cond])*psi_kz[cond]

            # vel_kx = adot*interp_fk(kmag)*psi_kx
            # vel_ky = adot*interp_fk(kmag)*psi_ky
            # vel_kz = adot*interp_fk(kmag)*psi_kz
            #
            # cond = np.where(kmag == 0.)
            # vel_kx[cond[0], cond[1], cond[2]] = 0 + 0*1j
            # vel_ky[cond[0], cond[1], cond[2]] = 0 + 0*1j
            # vel_kz[cond[0], cond[1], cond[2]] = 0 + 0*1j

            vel_x = shift.cart.mpi_ifft3D(vel_kx, x_shape, boxsize, ngrid, FFT)
            vel_y = shift.cart.mpi_ifft3D(vel_ky, x_shape, boxsize, ngrid, FFT)
            vel_z = shift.cart.mpi_ifft3D(vel_kz, x_shape, boxsize, ngrid, FFT)

        MPI.wait()

        if MPI.rank == 0:
            printout("Write ICs", 1, verbose)

        pos_x = pos_x.flatten()
        pos_y = pos_y.flatten()
        pos_z = pos_z.flatten()

        vel_x = vel_x.flatten()
        vel_y = vel_y.flatten()
        vel_z = vel_z.flatten()

        part_len = np.array([len(pos_x)])
        part_lens = MPI.collect(part_len)

        if MPI.rank == 0:
            part_id_offsets = np.cumsum(part_lens)
            MPI.send(part_id_offsets, tag=11)
        else:
            part_id_offsets = MPI.recv(0, tag=11)

        MPI.wait()

        part_id_offsets = np.array([0] + np.ndarray.tolist(part_id_offsets))

        H0 = params["Cosmology"]["H0"]
        Omega_m = params["Cosmology"]["Omega_m"]
        boxsize = params["Siminfo"]["Boxsize"]
        Ngrid = params["Siminfo"]["Ngrid"]
        G_const = 6.6743e-11
        part_mass = 3.*Omega_m*(boxsize**3.)/(8.*np.pi*G_const*Ngrid**3.)
        part_mass *= 3.0857e2/1.9891
        part_mass /= 1e10

        if MPI.rank == 0:
            printout("Particle mass = "+str(part_mass), 2, verbose)

        if params["ICs"]["Single"] is True:

            header = {
              'nfiles'        : 1,
              'massarr'       : part_mass,
              'npart_all'     : Ngrid**3,
              'time'          : mimic.theory.z2a(z_init),
              'redshift'      : z_init,
              'boxsize'       : boxsize,
              'omegam'        : Omega_m,
              'omegal'        : 1.-Omega_m,
              'hubble'        : H0*1e-2
            }

            pos_x = MPI.collect(pos_x)
            pos_y = MPI.collect(pos_y)
            pos_z = MPI.collect(pos_z)
            vel_x = MPI.collect(vel_x)
            vel_y = MPI.collect(vel_y)
            vel_z = MPI.collect(vel_z)

        else:

            header = {
              'nfiles'        : MPI.size,
              'massarr'       : part_mass,
              'npart_all'     : Ngrid**3,
              'time'          : mimic.theory.z2a(z_init),
              'redshift'      : z_init,
              'boxsize'       : boxsize,
              'omegam'        : Omega_m,
              'omegal'        : 1.-Omega_m,
              'hubble'        : H0*1e-2
            }

        pos = np.column_stack([pos_x, pos_y, pos_z])
        vel = np.column_stack([vel_x, vel_y, vel_z])

        printout("Processor - " + str(MPI.rank) + " particle position shape " + str(np.shape(pos)), 1, verbose)
        printout("Processor - " + str(MPI.rank) + " particle velocity shape " + str(np.shape(vel)), 1, verbose)

        if params["ICs"]["Single"] is False:
            fname = params['Siminfo']['Simname'] + '_IC_'+str(MPI.rank)+'.npz'
            np.savez(fname, pos=pos, vel=vel)

            fname = params['Siminfo']['Simname'] + '_IC.'+str(MPI.rank)
            mimic.write.write_gadget(fname, header, pos, vel, ic_format=2, single=True,
                                     id_offset=part_id_offsets[MPI.rank])
        else:
            if MPI.rank == 0:
                fname = params['Siminfo']['Simname'] + '_IC_'+str(MPI.rank)+'.npz'
                np.savez(fname, pos=pos, vel=vel)

                fname = params['Siminfo']['Simname'] + '_IC.'+str(MPI.rank)
                mimic.write.write_gadget(fname, header, pos, vel, ic_format=1, single=True,
                                         id_offset=part_id_offsets[MPI.rank])

if MPI.rank == 0:
    printspace(verbose)
    printout("> COMPLETE ---------------------------------------------------------<", 0, verbose)
    printout("=====================================================================", 0, verbose)

MPI.end()
