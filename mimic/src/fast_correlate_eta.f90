include "fast_correlate.f90"


subroutine corr_dot_eta(x1, xc, y1, yc, z1, zc, ex1, exc, ey1, eyc, ez1, ezc &
  , type1, typec, adot_phi, adot_vel, logr, xi, zeta_p, zeta_u, psir_pp, psit_pp &
  , psir_pu, psit_pu, psir_uu, psit_uu, boxsize, lenr, lenx1, lenxc, eta, field &
  , mpi_rank, lenpro, lenpre, prefix)

  ! Outputs will only be for a single type, aka type1.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: lenr, lenx1, lenxc
  integer, intent(in) :: type1, typec(lenxc)
  real(kind=dp), intent(in) :: x1(lenx1), xc(lenxc), y1(lenx1), yc(lenxc), z1(lenx1), zc(lenxc)
  real(kind=dp), intent(in) :: ex1, exc(lenxc), ey1, eyc(lenxc), ez1, ezc(lenxc), eta(lenxc)
  real(kind=dp), intent(in) :: adot_phi, adot_vel, logr(lenr), xi(lenr), zeta_u(lenr), zeta_p(lenr)
  real(kind=dp), intent(in) :: psir_pp(lenr), psit_pp(lenr), psir_pu(lenr), psit_pu(lenr)
  real(kind=dp), intent(in) :: psir_uu(lenr), psit_uu(lenr), boxsize
  real(kind=dp), intent(out) :: field(lenx1)

  integer, intent(in) :: mpi_rank, lenpro, lenpre
  character, intent(in) :: prefix(lenpre)

  integer :: i, j
  real(kind=dp) :: cc(lenxc), field_val

  do i=1, lenx1
    call get_cc_array2(x1(i), xc, y1(i), yc, z1(i), zc, ex1, exc, ey1, eyc, ez1, ezc &
      , type1, typec, adot_phi, adot_vel, logr, xi, zeta_p, zeta_u, psir_pp, psit_pp &
      , psir_pu, psit_pu, psir_uu, psit_uu, boxsize, lenr, lenxc, cc)
    field_val = 0.
    do j=1, lenxc
      field_val = field_val + cc(j)*eta(j)
    end do
    field(i) = field_val
    if (mpi_rank .eq. 0) then
      call progress_bar(i, lenx1, lenpro, prefix, lenpre)
    end if
  end do

end subroutine corr_dot_eta
