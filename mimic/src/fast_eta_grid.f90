include "interp.f90"
include "progress.f90"


subroutine periodic_1d_single(rx, boxsize, newrx)

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

end subroutine periodic_1d_single


subroutine periodic_3d_single(rx, ry, rz, boxsize, newrx, newry, newrz)

  implicit none
  integer, parameter :: dp = kind(1.d0)

  real(kind=dp), intent(in) :: rx, ry, rz, boxsize
  real(kind=dp), intent(out) :: newrx, newry, newrz

  call periodic_1d_single(rx, boxsize, newrx)
  call periodic_1d_single(ry, boxsize, newry)
  call periodic_1d_single(rz, boxsize, newrz)

end subroutine periodic_3d_single


subroutine snap2grid(x, dx, newx)

  implicit none
  integer, parameter :: dp = kind(1.d0)

  real(kind=dp), intent(in) :: x, dx
  real(kind=dp), intent(out) :: newx
  real(kind=dp) :: fx

  fx = x/dx
  newx = real(nint(fx))*dx

end subroutine snap2grid


subroutine get_wf_grid_single_fast(cons_x, cons_y, cons_z, cons_ex, cons_ey, cons_ez &
  , cons_len, eta, x, y, z, adot, logr, zeta, lenzeta, boxsize, ngrid, val)

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: cons_len, lenzeta, ngrid
  real(kind=dp), intent(in) :: x, y, z, adot
  real(kind=dp), intent(in) :: cons_x(cons_len), cons_y(cons_len), cons_z(cons_len)
  real(kind=dp), intent(in) :: cons_ex(cons_len), cons_ey(cons_len), cons_ez(cons_len)
  real(kind=dp), intent(in) :: eta(cons_len)
  real(kind=dp), intent(in) :: logr(lenzeta), zeta(lenzeta), boxsize
  real(kind=dp), intent(out) :: val

  integer :: i
  real(kind=dp) :: rxo, ryo, rzo, rx, ry, rz, r
  real(kind=dp) :: nrx, nry, nrz, cov_du, du, dx

  dx = boxsize/real(ngrid)
  val = 0.
  do i = 1, cons_len
    rxo = cons_x(i) - x
    ryo = cons_y(i) - y
    rzo = cons_z(i) - z
    call periodic_3d_single(rxo, ryo, rzo, boxsize, rx, ry, rz)
    r = sqrt(rx**2 + ry**2 + rz**2)
    nrx = rx/r
    nry = ry/r
    nrz = rz/r
    call interp_log_float(logr, zeta, lenzeta, log10(r), zeta(1), zeta(lenzeta), cov_du)
    du = - adot*cov_du*nrx*cons_ex(i) - adot*cov_du*nry*cons_ey(i) - adot*cov_du*nrz*cons_ez(i)
    val = val + du*eta(i)
  end do

end subroutine get_wf_grid_single_fast


subroutine get_wf_grid_fast(cons_x, cons_y, cons_z, cons_ex, cons_ey, cons_ez &
  , cons_len, eta, x, y, z, lenx, adot, logr, zeta, lenzeta, prefix, lenpre&
  , lenpro, mpi_rank, boxsize, ngrid, values)

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: cons_len, lenzeta, lenx, lenpre, lenpro, mpi_rank, ngrid
  real(kind=dp), intent(in) :: x(lenx), y(lenx), z(lenx), adot
  real(kind=dp), intent(in) :: cons_x(cons_len), cons_y(cons_len), cons_z(cons_len)
  real(kind=dp), intent(in) :: cons_ex(cons_len), cons_ey(cons_len), cons_ez(cons_len)
  real(kind=dp), intent(in) :: eta(cons_len)
  real(kind=dp), intent(in) :: logr(lenzeta), zeta(lenzeta), boxsize
  character, intent(in) :: prefix(lenpre)
  real(kind=dp), intent(out) :: values(lenx)

  integer :: i

  do i = 1, lenx
    call get_wf_grid_single_fast(cons_x, cons_y, cons_z, cons_ex, cons_ey, cons_ez &
      , cons_len, eta, x(i), y(i), z(i), adot, logr, zeta, lenzeta, boxsize, ngrid, values(i))
    if (mpi_rank .eq. 0) then
      call progress_bar(i, lenx, lenpro, prefix, lenpre)
    end if
  end do

end subroutine get_wf_grid_fast
