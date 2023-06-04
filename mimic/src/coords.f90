
subroutine distance_1d_float(rx, boxsize, newrx)

  ! Distance in 1D.
  !
  ! Parameters
  ! ----------
  ! rx : float
  !   Radial distance.
  ! boxsize : float
  !   Boxsize.
  !
  ! Returns
  ! -------
  ! newrx : float
  !   Distance with periodic conditions.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  real(kind=dp), intent(in) :: rx, boxsize
  real(kind=dp), intent(out) :: newrx

  if (rx .lt. -boxsize/2.) then
    newrx = rx + boxsize
  else if (rx .gt. boxsize/2.) then
    newrx = rx - boxsize
  else
    newrx = rx
  end if
  newrx = rx

end subroutine distance_1d_float


subroutine distance_3d_float(rx, ry, rz, boxsize, newr, newrx, newry, newrz)

  ! Distance in 3D.
  !
  ! Parameters
  ! ----------
  ! rx, ry, rz : float
  !   Radial distance in each axis.
  ! boxsize : float
  !   Boxsize.
  !
  ! Returns
  ! -------
  ! newr : float
  !   Distance with periodic conditions.
  ! newrx, newry, newrz : float
  !   Distance with periodic conditions in each axes.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  real(kind=dp), intent(in) :: rx, ry, rz, boxsize
  real(kind=dp), intent(out) :: newr, newrx, newry, newrz

  call distance_1d_float(rx, boxsize, newrx)
  call distance_1d_float(ry, boxsize, newry)
  call distance_1d_float(rz, boxsize, newrz)

  newr = sqrt(newrx**2 + newry**2 + newrz**2)

end subroutine distance_3d_float


subroutine get_vec_norm_float(x, y, z, nx, ny, nz)

  ! Normalise vector.
  !
  ! Parameters
  ! ----------
  ! x, y, z : float
  !   Vector.
  !
  ! Returns
  ! -------
  ! nx, ny, nz : float
  !   Normalised vector.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  real(kind=dp), intent(in) :: x, y, z
  real(kind=dp), intent(out) :: nx, ny, nz

  real(kind=dp) :: r

  r = sqrt(x**2 + y**2 + z**2)

  if (r .eq. 0) then
    nx = 1./sqrt(3.)
    ny = 1./sqrt(3.)
    nz = 1./sqrt(3.)
  else
    nx = x/r
    ny = y/r
    nz = z/r
  end if

end subroutine get_vec_norm_float