include "fast_correlate.f90"


subroutine corr_dot_eta(x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2 &
  , type1, type2, adot_phi, adot_vel, logr, xi, zeta_p, zeta_u, psir_pp, psit_pp &
  , psir_pu, psit_pu, psir_uu, psit_uu, boxsize, lenr, lenx1, lenx2, eta, field &
  , mpi_rank, lenpro, lenpre, prefix)

  ! Outputs will only be for a single type, aka type1.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: lenr, lenx1, lenx2
  integer, intent(in) :: type1, type2(lenx2)
  real(kind=dp), intent(in) :: x1(lenx1), x2(lenx2), y1(lenx1), y2(lenx2), z1(lenx1), z2(lenx2)
  real(kind=dp), intent(in) :: ex1, ex2(lenx2), ey1, ey2(lenx2), ez1, ez2(lenx2), eta(lenx2)
  real(kind=dp), intent(in) :: adot_phi, adot_vel, logr(lenr), xi(lenr), zeta_u(lenr), zeta_p(lenr)
  real(kind=dp), intent(in) :: psir_pp(lenr), psit_pp(lenr), psir_pu(lenr), psit_pu(lenr)
  real(kind=dp), intent(in) :: psir_uu(lenr), psit_uu(lenr), boxsize
  real(kind=dp), intent(out) :: field(lenx1)

  integer, intent(in) :: mpi_rank, lenpro, lenpre
  character, intent(in) :: prefix(lenpre)

  integer :: i, j
  real(kind=dp) :: cc(lenx2), field_val

  do i=1, lenx1
    call get_cc_array2(x1(i), x2, y1(i), y2, z1(i), z2, ex1, ex2, ey1, ey2, ez1, ez2 &
      , type1, type2, adot_phi, adot_vel, logr, xi, zeta_p, zeta_u, psir_pp, psit_pp &
      , psir_pu, psit_pu, psir_uu, psit_uu, boxsize, lenr, lenx2, cc)
    field_val = 0.
    do j=1, lenx2
      field_val = field_val + cc(j)*eta(j)
    end do
    field(i) = field_val
    if (mpi_rank .eq. 0) then
      call progress_bar(i, lenx1, lenpro, prefix, lenpre)
    end if
  end do

end subroutine corr_dot_eta
