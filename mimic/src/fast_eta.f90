include "interp.f90"
include "progress.f90"


subroutine get_wf_single_fast(cons_x, cons_y, cons_z, cons_ex, cons_ey, cons_ez &
  , cons_len, eta, x, y, z, adot, logr, zeta, lenzeta, val)

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: cons_len, lenzeta
  real(kind=dp), intent(in) :: x, y, z, adot
  real(kind=dp), intent(in) :: cons_x(cons_len), cons_y(cons_len), cons_z(cons_len)
  real(kind=dp), intent(in) :: cons_ex(cons_len), cons_ey(cons_len), cons_ez(cons_len)
  real(kind=dp), intent(in) :: eta(cons_len)
  real(kind=dp), intent(in) :: logr(lenzeta), zeta(lenzeta)
  real(kind=dp), intent(out) :: val

  integer :: i
  real(kind=dp) :: rx, ry, rz, r, nrx, nry, nrz, cov_du, du

  val = 0.
  do i = 1, cons_len
    rx = cons_x(i) - x
    ry = cons_y(i) - y
    rz = cons_z(i) - z
    r = sqrt(rx**2 + ry**2 + rz**2)
    nrx = rx/r
    nry = ry/r
    nrz = rz/r
    call interp_log_float(logr, zeta, lenzeta, log(r), zeta(1), zeta(lenzeta), cov_du)
    du = - adot*cov_du*nrx*cons_ex(i) - adot*cov_du*nry*cons_ey(i) - adot*cov_du*nrz*cons_ez(i)
    val = val + du*eta(i)
  end do

end subroutine get_wf_single_fast


subroutine get_wf_fast(cons_x, cons_y, cons_z, cons_ex, cons_ey, cons_ez &
  , cons_len, eta, x, y, z, lenx, adot, logr, zeta, lenzeta, prefix, lenpre&
  , lenpro, mpi_rank, values)

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: cons_len, lenzeta, lenx, lenpre, lenpro, mpi_rank
  real(kind=dp), intent(in) :: x(lenx), y(lenx), z(lenx), adot
  real(kind=dp), intent(in) :: cons_x(cons_len), cons_y(cons_len), cons_z(cons_len)
  real(kind=dp), intent(in) :: cons_ex(cons_len), cons_ey(cons_len), cons_ez(cons_len)
  real(kind=dp), intent(in) :: eta(cons_len)
  real(kind=dp), intent(in) :: logr(lenzeta), zeta(lenzeta)
  character, intent(in) :: prefix(lenpre)
  real(kind=dp), intent(out) :: values(lenx)

  integer :: i

  do i = 1, lenx
    call get_wf_single_fast(cons_x, cons_y, cons_z, cons_ex, cons_ey, cons_ez &
      , cons_len, eta, x(i), y(i), z(i), adot, logr, zeta, lenzeta, values(i))
    if (mpi_rank .eq. 0) then
      call progress_bar(i, lenx, lenpro, prefix, lenpre)
    end if
  end do

end subroutine get_wf_fast
