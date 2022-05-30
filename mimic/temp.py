import numpy as np
from scipy.interpolate import interp1d
import mpiutils
from mpi4py_fft import PFFT, newDistArray
import shift
import pylustre
import magpie

MPI = mpiutils.MPI()

path = '/lustre/tetyda/home/knaidoo/projects/lustre/simulations/ICs/HighRes/'
path2CF2 = '/lustre/tetyda/home/knaidoo/projects/lustre/simulations/ICs/CF2/'

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

xL, yL, zL = shift.cart.grid3d(boxsize, 512)

#FFT = PFFT(MPI.comm, Ngrids, axes=(0, 1, 2), dtype=complex, grid=(-1,), transfo
rm='fftn')

data = np.loadtxt(path + 'cosmo/LCDM_FID/growth_lcdm_scale_independent.txt', unp
ack=True)
z_LCDM, D_LCDM, f_LCDM = data[0], data[1], data[2]

data = np.loadtxt(path + 'cosmo/nDGP_N1/growth_MG_scale_independent.txt', unpack
=True)
z_NDGP, D_NDGP, f_NDGP = data[0], data[1], data[2]

func_D_LCDM = interp1d(z_LCDM, D_LCDM, kind='cubic')
func_D_NDGP = interp1d(z_NDGP, D_NDGP, kind='cubic')

for i in range(1, 5+1):

    if MPI.rank == 0:
        MPI.mpi_print('Create whitenoise for R'+str(i))

    data = np.load(path + 'whitenoise/R'+str(i)+'/whitenoise_real_space_'+str(MP
I.rank)+'.npz')
    x3d = data['x3d']
    y3d = data['y3d']
    z3d = data['z3d']
    #w = data['w']

    xmin = x3d.min() - dx/2.
    xmax = x3d.max() + dx/2.
    ymin = y3d.min() - dx/2.
    ymax = y3d.max() + dx/2.
    zmin = z3d.min() - dx/2.
    zmax = z3d.max() + dx/2.

    originout = [xmin, ymin, zmin]
    boxsizeout = [xmax-xmin, ymax-ymin, zmax-zmin]

    data = np.load(path + 'whitenoise/R'+str(i)+'/whitenoise_fourier_space_'+str
(MPI.rank)+'.npz')
    kx3d = data['kx3d']
    ky3d = data['ky3d']
    kz3d = data['kz3d']
    #wk = data['wk']

    fname = path2CF2 + 'LCDM/FID/R'+str(i)+'/N512B500/density_N512B500_512_500_5
00'+str(i)+'.dat'

    dgrid = pylustre.icecore.load_dgrid(fname, 512)
    dgrid = pylustre.icecore.rearrange_dgrid(dgrid)

    dens = magpie.remap.grid2grid3d(dgrid, boxsize, np.array(np.shape(x3d)),
                                    origin=0.0, originout=originout,
                                    boxsizeout=boxsizeout)

    np.savez(path + 'CF2/LCDM/FID/R'+str(i) + '/dens_LowRes_z0_'+str(MPI.rank)+'
.npz',
             x3d=x3d, y3d=y3d, z3d=z3d, dens=dens)

    dens *= func_D_LCDM(127.)

    np.savez(path + 'CF2/LCDM/FID/R'+str(i) + '/dens_LowRes_z127_'+str(MPI.rank)
+'.npz',
             x3d=x3d, y3d=y3d, z3d=z3d, dens=dens)

    fname = path2CF2 + 'nDGP/N1/R'+str(i)+'/N512B500/density_N512B500_512_500_50
0'+str(i)+'.dat'

    dgrid = pylustre.icecore.load_dgrid(fname, 512)
    dgrid = pylustre.icecore.rearrange_dgrid(dgrid)

    dens = magpie.remap.grid2grid3d(dgrid, boxsize, np.array(np.shape(x3d)),
                                    origin=0.0, originout=originout,
                                    boxsizeout=boxsizeout)

    np.savez(path + 'CF2/nDGP/N1/R'+str(i) + '/dens_LowRes_z0_'+str(MPI.rank)+'.
npz',
             x3d=x3d, y3d=y3d, z3d=z3d, dens=dens)

    dens *= func_D_LCDM(127.)

    np.savez(path + 'CF2/nDGP/N1/R'+str(i) + '/dens_LowRes_z127_'+str(MPI.rank)+
'.npz',
             x3d=x3d, y3d=y3d, z3d=z3d, dens=dens)

    if MPI.rank == 0:
        MPI.mpi_print('Finished R'+str(i))
