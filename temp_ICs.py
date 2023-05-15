import numpy as np
from scipy.interpolate import interp1d
import mpiutils
from mpi4py_fft import PFFT, newDistArray
import shift
import pylustre
import magpie

def Hz(z, H0=67.66, omega_m=0.3111, omega_l=1.-0.3111):
    return H0*np.sqrt(omega_m*(1.+z)**3. + omega_l)

def adot(z):
    a = 1/(1+z)
    return a*Hz(z)

MPI = mpiutils.MPI()

path = '/lustre/tetyda/home/knaidoo/projects/lustre/simulations/ICs/HighRes/'

data = np.loadtxt(path + 'cosmo/LCDM_FID/growth_lcdm_scale_independent.txt', unpack=True)
z_LCDM, D_LCDM, f_LCDM = data[0], data[1], data[2]

data = np.loadtxt(path + 'cosmo/nDGP_N1/growth_MG_scale_independent.txt', unpack=True)
z_NDGP, D_NDGP, f_NDGP = data[0], data[1], data[2]

func_f_LCDM = interp1d(z_LCDM, f_LCDM, kind='cubic')
func_f_NDGP = interp1d(z_NDGP, f_NDGP, kind='cubic')

ngrid = 2048
boxsize = 500.
dx = boxsize / ngrid
norm = (dx/np.sqrt(2.*np.pi))**3.

Ngrids = np.array([ngrid, ngrid, ngrid], dtype=int)

xedges, x = shift.cart.grid1d(boxsize, ngrid)
y = np.copy(x)
z = np.copy(x)

kx = shift.cart.get_fourier_grid_1D(boxsize, ngrid)
ky = np.copy(kx)
kz = np.copy(kx)

FFT = PFFT(MPI.comm, Ngrids, axes=(0, 1, 2), dtype=complex, grid=(-1,), transform='fftn')
A = newDistArray(FFT, False)

models = ['LCDM/FID/', 'nDGP/N1/']

for i in range(1, 5+1):

    for model in models:

        if MPI.rank == 0:
            MPI.mpi_print('Construct HighRes R'+str(i), 'for model', model)
            MPI.mpi_print('---> Load LowRes CF2')

        data = np.load(path + 'CF2/'+model+'R'+str(i) + '/dens_HighRes_z127_'+str(MPI.rank)+'.npz')
        x3d = data['x3d']
        y3d = data['y3d']
        z3d = data['z3d']
        dens = data['dens']

        if MPI.rank == 0:
            MPI.mpi_print('---> Forward FFT')

        A = dens + 1j*np.zeros(np.shape(A))
        Ak = FFT.forward(A, normalize=False)
        Ak *= norm

        Ak_part = MPI.check_partition(Ngrids, np.array(np.shape(Ak)))
        kx3d, ky3d, kz3d = MPI.create_split_ndgrid([kx, ky, kz], Ak_part)

        k = np.sqrt(kx3d**2. + ky3d**2. + kz3d**2.)

        if MPI.rank == 0:
            MPI.mpi_print('---> Store Fourier modes')

        np.savez(path + 'CF2/'+model+'R'+str(i) + '/densk_HighRes_z127_'+str(MPI.rank)+'.npz', kx3d=kx3d, ky3d=ky3d, kz3d=kz3d, densk=Ak)

        if MPI.rank == 0:
            MPI.mpi_print('---> Compute displacement field')

        Axk = Ak*1j*kx3d/(k**2.)
        Ayk = Ak*1j*ky3d/(k**2.)
        Azk = Ak*1j*kz3d/(k**2.)

        cond = np.where(k == 0.)
        Axk[cond] = 0 + 0*1j
        Ayk[cond] = 0 + 0*1j
        Azk[cond] = 0 + 0*1j

        Ax = np.zeros_like(A)
        Ay = np.zeros_like(A)
        Az = np.zeros_like(A)

        Ax = FFT.backward(Axk, Ax, normalize=True)
        Ay = FFT.backward(Ayk, Ay, normalize=True)
        Az = FFT.backward(Azk, Az, normalize=True)

        Ax /= norm
        Ay /= norm
        Az /= norm

        phix = Ax.real
        phiy = Ay.real
        phiz = Az.real

        if MPI.rank == 0:
            MPI.mpi_print('---> Compute velocities')

        if model == 'LCDM/FID/':
            f_growth = func_f_LCDM(127.)
        elif model == 'nDGP/N1/':
            f_growth = func_f_NDGP(127.)

        ux = adot(127.)*f_growth*phix
        uy = adot(127.)*f_growth*phiy
        uz = adot(127.)*f_growth*phiz

        if MPI.rank == 0:
            MPI.mpi_print('---> Save particle positions and velocities')

        np.savez(path + 'CF2/'+model+'R'+str(i) + '/particle_IC_z127_'+str(MPI.rank)+'.npz',
                 x3d=x3d, y3d=y3d, z3d=z3d, phix=phix, phiy=phiy, phiz=phiz, ux=ux, uy=uy, uz=uz)

        if MPI.rank == 0:
            MPI.mpi_print('Finished R'+str(i), 'for model', model)
