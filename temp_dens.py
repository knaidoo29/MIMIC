
import numpy as np
from scipy.interpolate import interp1d
import mpiutils
from mpi4py_fft import PFFT, newDistArray
import shift
import pylustre
import magpie

kL_nyq = shift.cart.get_kn(500., 512)

def get_FL(k, k0=0.5*kL_nyq, T=0.1):
    return 1./(np.exp((k-k0)/(k0*T)) + 1.)

def get_FH(k, k0=0.5*kL_nyq, T=0.1):
    return np.sqrt(1. - get_FL(k, k0=k0, T=T)**2.)

MPI = mpiutils.MPI()

path = '/lustre/tetyda/home/knaidoo/projects/lustre/simulations/ICs/HighRes/'

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

        data = np.load(path + 'CF2/'+model+'R'+str(i) + '/dens_LowRes_z127_'+str(MPI.rank)+'.npz')
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
        create_split_ndgrid(self, arrays_nd, whichaxis)

        k = np.sqrt(kx3d**2. + ky3d**2. + kz3d**2.)

        if MPI.rank == 0:
            MPI.mpi_print('---> Store Fourier modes')

        np.savez(path + 'CF2/'+model+'R'+str(i) + '/densk_z127_'+str(MPI.rank)+'.npz', kx3d=kx3d, ky3d=ky3d, kz3d=kz3d, densk=Ak)

        if MPI.rank == 0:
            MPI.mpi_print('---> Apply low Fourier filter')

        Ak *= get_FL(k)

        if MPI.rank == 0:
            MPI.mpi_print('---> Store Fourier modes')

        np.savez(path + 'CF2/'+model+'R'+str(i) + '/densk_FilterLow_z127_'+str(MPI.rank)+'.npz', kx3d=kx3d, ky3d=ky3d, kz3d=kz3d, densk=Ak)

        if MPI.rank == 0:
            MPI.mpi_print('---> Backward FFT')

        Anew = np.zeros_like(A)
        Anew = FFT.backward(Ak, Anew, normalize=True)
        Anew /= norm

        if MPI.rank == 0:
            MPI.mpi_print('---> Store real dens with low Fourier filter')

        np.savez(path + 'CF2/'+model+'R'+str(i) + '/dens_FilterLow_z127_'+str(MPI.rank)+'.npz', x3d=x3d, y3d=y3d, z3d=z3d, dens=Anew.real)

        if MPI.rank == 0:
            MPI.mpi_print('---> Load HighRes CF2')

        data = np.load(path + 'rand/'+model+'R'+str(i) + '/dens_z127_'+str(MPI.rank)+'.npz')
        dens = data['dens']

        if MPI.rank == 0:
            MPI.mpi_print('---> Forward FFT')

        A = dens + 1j*np.zeros(np.shape(A))
        Ak = FFT.forward(A, normalize=False)
        Ak *= norm

        if MPI.rank == 0:
            MPI.mpi_print('---> Store Fourier modes')

        np.savez(path + 'rand/'+model+'R'+str(i) + '/densk_z127_'+str(MPI.rank)+'.npz', kx3d=kx3d, ky3d=ky3d, kz3d=kz3d, densk=Ak)

        if MPI.rank == 0:
            MPI.mpi_print('---> Apply high Fourier filter')

        Ak *= get_FH(k)

        if MPI.rank == 0:
            MPI.mpi_print('---> Store Fourier modes')

        np.savez(path + 'CF2/'+model+'R'+str(i) + '/densk_FilterHigh_z127_'+str(MPI.rank)+'.npz', kx3d=kx3d, ky3d=ky3d, kz3d=kz3d, densk=Ak)

        if MPI.rank == 0:
            MPI.mpi_print('---> Backward FFT')

        Anew = np.zeros_like(A)
        Anew = FFT.backward(Ak, Anew, normalize=True)
        Anew /= norm

        if MPI.rank == 0:
            MPI.mpi_print('---> Store real dens with high Fourier filter')

        np.savez(path + 'CF2/'+model+'R'+str(i) + '/dens_FilterHigh_z127_'+str(MPI.rank)+'.npz', x3d=x3d, y3d=y3d, z3d=z3d, dens=Anew.real)

        if MPI.rank == 0:
            MPI.mpi_print('---> Add real CF2 dens with low Fourier filter to real rand dens with high Fourier filter')

        dataL = np.load(path + 'CF2/'+model+'R'+str(i) + '/dens_FilterLow_z127_'+str(MPI.rank)+'.npz')
        dataH = np.load(path + 'CF2/'+model+'R'+str(i) + '/dens_FilterHigh_z127_'+str(MPI.rank)+'.npz')
        dens = dataL['dens'] + dataH['dens']

        if MPI.rank == 0:
            MPI.mpi_print('---> Store high res CF2 dens')

        np.savez(path + 'CF2/'+model+'R'+str(i) + '/dens_HighRes_z127_'+str(MPI.rank)+'.npz', x3d=x3d, y3d=y3d, z3d=z3d, dens=dens)

        if MPI.rank == 0:
            MPI.mpi_print('Finished R'+str(i), 'for model', model)
