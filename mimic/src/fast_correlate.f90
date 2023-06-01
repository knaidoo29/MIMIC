include "interp.f90"
include "progress.f90"
include "coords.f90"


subroutine get_dd_float(x1, x2, y1, y2, z1, z2, logr, xi, lenr, boxsize, dd_val)

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: lenr
  real(kind=dp), intent(in) :: logr(lenr), xi(lenr)
  real(kind=dp), intent(in) :: x1, x2, y1, y2, z1, z2, boxsize
  real(kind=dp), intent(out) :: dd_val

  real(kind=dp) :: rx, ry, rz, newr, newrx, newry, newrz

  rx = x2-x1
  ry = y2-y1
  rz = z2-z1

  call distance_3d_float(rx, ry, rz, boxsize, newr, newrx, newry, newrz)

  call interp_log_float(logr, xi, lenr, log10(newr), xi(1), xi(lenr), dd_val)

end subroutine get_dd_float


subroutine get_dp_float(x1, x2, y1, y2, z1, z2, ex, ey, ez, adot, logr, zeta, lenr &
  , boxsize, dp_val)

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: lenr
  real(kind=dp), intent(in) :: logr(lenr), zeta(lenr)
  real(kind=dp), intent(in) :: x1, x2, y1, y2, z1, z2, ex, ey, ez, adot, boxsize
  real(kind=dp), intent(out) :: dp_val

  real(kind=dp) :: rx, ry, rz, newr, newrx, newry, newrz, nrx, nry, nrz
  real(kind=dp) :: nex, ney, nez, zeta_val

  rx = x2-x1
  ry = y2-y1
  rz = z2-z1

  call distance_3d_float(rx, ry, rz, boxsize, newr, newrx, newry, newrz)
  call get_vec_norm_float(newrx, newry, newrz, nrx, nry, nrz)

  ! call get_vec_norm_float(ex, ey, ez, nex, ney, nez)
  nex = ex
  ney = ey
  nez = ez

  call interp_log_float(logr, zeta, lenr, log10(newr), zeta(1), zeta(lenr), zeta_val)

  dp_val = -adot*zeta_val*(nex*nrx + ney*nry + nez*nrz)

end subroutine get_dp_float


subroutine get_pp_float(x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2 &
  , adot2, logr, psir, psit, lenr, boxsize, pp_val)

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: lenr
  real(kind=dp), intent(in) :: logr(lenr), psir(lenr), psit(lenr)
  real(kind=dp), intent(in) :: x1, x2, y1, y2, z1, z2
  real(kind=dp), intent(in) :: ex1, ex2, ey1, ey2, ez1, ez2, adot2, boxsize
  real(kind=dp), intent(out) :: pp_val

  real(kind=dp) :: rx, ry, rz, newr, newrx, newry, newrz, nrx, nry, nrz
  real(kind=dp) :: pr, pt, p1, p2, pp_x, pp_y, pp_z
  real(kind=dp) :: pp_xx, pp_xy, pp_xz, pp_yx, pp_yy, pp_yz, pp_zx, pp_zy, pp_zz

  rx = x2-x1
  ry = y2-y1
  rz = z2-z1

  call distance_3d_float(rx, ry, rz, boxsize, newr, newrx, newry, newrz)
  call get_vec_norm_float(newrx, newry, newrz, nrx, nry, nrz)

  call interp_log_float(logr, psir, lenr, log10(newr), psir(1), psir(lenr), pr)
  call interp_log_float(logr, psit, lenr, log10(newr), psit(1), psit(lenr), pt)

  p1 = pt
  p2 = pr-pt

  pp_xx = p1+p2*nrx*nrx
  pp_yy = p1+p2*nry*nry
  pp_zz = p1+p2*nrz*nrz

  pp_xy = p2*nrx*nry
  pp_xz = p2*nrx*nrz
  pp_yx = p2*nry*nrx
  pp_yz = p2*nry*nrz
  pp_zx = p2*nrz*nrx
  pp_zy = p2*nrz*nry

  pp_x = ex1*(pp_xx*ex2 + pp_xy*ey2 + pp_xz*ez2)
  pp_y = ey1*(pp_yx*ex2 + pp_yy*ey2 + pp_yz*ez2)
  pp_z = ez1*(pp_zx*ex2 + pp_zy*ey2 + pp_zz*ez2)
  pp_val = adot2*(pp_x + pp_y + pp_z)

  if (newr .eq. 0) then
    pp_val = adot2*pt*(ex1*ex2 + ey1*ey2 + ez1*ez2)
  end if

end subroutine get_pp_float


subroutine get_cc_float(x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2 &
  , type1, type2, adot_phi, adot_vel, logr, xi, zeta_p, zeta_u, psir_pp, psit_pp &
  , psir_pu, psit_pu, psir_uu, psit_uu, boxsize, lenr, cc)

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: lenr, type1, type2
  real(kind=dp), intent(in) :: x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2
  real(kind=dp), intent(in) :: adot_phi, adot_vel, logr(lenr), xi(lenr), zeta_p(lenr)
  real(kind=dp), intent(in) :: zeta_u(lenr), psir_pp(lenr), psit_pp(lenr), psir_pu(lenr)
  real(kind=dp), intent(in) :: psit_pu(lenr), psir_uu(lenr), psit_uu(lenr), boxsize
  real(kind=dp), intent(out) :: cc

  if ((type1 .eq. 0) .and. (type2 .eq. 0)) then
    call get_dd_float(x1, x2, y1, y2, z1, z2, logr, xi, lenr, boxsize, cc)
  end if

  if ((type1 .eq. 1) .and. (type2 .eq. 1)) then
    call get_pp_float(x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2 &
      , adot_phi*adot_phi, logr, psir_pp, psit_pp, lenr, boxsize, cc)
  end if

  if ((type1 .eq. 2) .and. (type2 .eq. 2)) then
    call get_pp_float(x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2 &
      , adot_vel*adot_vel, logr, psir_uu, psit_uu, lenr, boxsize, cc)
  end if

  if ((type1 .eq. 0) .and. (type2 .eq. 1)) then
    call get_dp_float(x1, x2, y1, y2, z1, z2, ex2, ey2, ez2, adot_phi, logr &
      , zeta_p, lenr, boxsize, cc)
  end if

  if ((type1 .eq. 1) .and. (type2 .eq. 0)) then
    call get_dp_float(x1, x2, y1, y2, z1, z2, ex1, ey1, ez1, adot_phi, logr &
      , zeta_p, lenr, boxsize, cc)
  end if

  if ((type1 .eq. 0) .and. (type2 .eq. 2)) then
    call get_dp_float(x1, x2, y1, y2, z1, z2, ex2, ey2, ez2, adot_vel, logr &
      , zeta_u, lenr, boxsize, cc)
  end if

  if ((type1 .eq. 2) .and. (type2 .eq. 0)) then
    call get_dp_float(x1, x2, y1, y2, z1, z2, ex1, ey1, ez1, adot_vel, logr &
      , zeta_u, lenr, boxsize, cc)
  end if

  if (((type1 .eq. 1) .and. (type2 .eq. 2)) .or. ((type1 .eq. 2) .and. (type2 .eq. 1))) then
    call get_pp_float(x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2 &
    , adot_phi*adot_vel, logr, psir_pu, psit_pu, lenr, boxsize, cc)
  end if

end subroutine get_cc_float


subroutine get_cc_array1(x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2 &
  , type1, type2, adot_phi, adot_vel, logr, xi, zeta_p, zeta_u, psir_pp, psit_pp &
  , psir_pu, psit_pu, psir_uu, psit_uu, boxsize, lenr, lenx1, cc)

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: lenr, lenx1
  integer, intent(in) :: type1(lenx1), type2
  real(kind=dp), intent(in) :: x1(lenx1), x2, y1(lenx1), y2, z1(lenx1), z2
  real(kind=dp), intent(in) :: ex1(lenx1), ex2, ey1(lenx1), ey2, ez1(lenx1), ez2
  real(kind=dp), intent(in) :: adot_phi, adot_vel, logr(lenr), xi(lenr), zeta_p(lenr)
  real(kind=dp), intent(in) :: zeta_u(lenr), psir_pp(lenr), psit_pp(lenr), psir_pu(lenr)
  real(kind=dp), intent(in) :: psit_pu(lenr), psir_uu(lenr), psit_uu(lenr), boxsize
  real(kind=dp), intent(out) :: cc(lenx1)

  integer :: i

  do i = 1, lenx1
    call get_cc_float(x1(i), x2, y1(i), y2, z1(i), z2, ex1(i), ex2, ey1(i), ey2 &
      , ez1(i), ez2, type1(i), type2, adot_phi, adot_vel, logr, xi, zeta_p, zeta_u &
      , psir_pp, psit_pp, psir_pu, psit_pu, psir_uu, psit_uu, boxsize, lenr, cc(i))
  end do

end subroutine get_cc_array1


subroutine get_cc_array2(x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2 &
  , type1, type2, adot_phi, adot_vel, logr, xi, zeta_p, zeta_u, psir_pp, psit_pp &
  , psir_pu, psit_pu, psir_uu, psit_uu, boxsize, lenr, lenx2, cc)

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: lenr, lenx2
  integer, intent(in) :: type1, type2(lenx2)
  real(kind=dp), intent(in) :: x1, x2(lenx2), y1, y2(lenx2), z1, z2(lenx2)
  real(kind=dp), intent(in) :: ex1, ex2(lenx2), ey1, ey2(lenx2), ez1, ez2(lenx2)
  real(kind=dp), intent(in) :: adot_phi, adot_vel, logr(lenr), xi(lenr), zeta_p(lenr)
  real(kind=dp), intent(in) :: zeta_u(lenr), psir_pp(lenr), psit_pp(lenr), psir_pu(lenr)
  real(kind=dp), intent(in) :: psit_pu(lenr), psir_uu(lenr), psit_uu(lenr), boxsize
  real(kind=dp), intent(out) :: cc(lenx2)

  integer :: i

  do i = 1, lenx2
    call get_cc_float(x1, x2(i), y1, y2(i), z1, z2(i), ex1, ex2(i), ey1, ey2(i) &
      , ez1, ez2(i), type1, type2(i), adot_phi, adot_vel, logr, xi, zeta_p, zeta_u &
      , psir_pp, psit_pp, psir_pu, psit_pu, psir_uu, psit_uu, boxsize, lenr, cc(i))
  end do

end subroutine get_cc_array2


subroutine get_cc_arrays(x1, x2, y1, y2, z1, z2, ex1, ex2, ey1, ey2, ez1, ez2 &
  , type1, type2, adot_phi, adot_vel, logr, xi, zeta_p, zeta_u, psir_pp, psit_pp &
  , psir_pu, psit_pu, psir_uu, psit_uu, boxsize, lenr, lenx, cc)

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: lenr, lenx
  integer, intent(in) :: type1(lenx), type2(lenx)
  real(kind=dp), intent(in) :: x1(lenx), x2(lenx), y1(lenx), y2(lenx), z1(lenx), z2(lenx)
  real(kind=dp), intent(in) :: ex1(lenx), ex2(lenx), ey1(lenx), ey2(lenx), ez1(lenx), ez2(lenx)
  real(kind=dp), intent(in) :: adot_phi, adot_vel, logr(lenr), xi(lenr), zeta_p(lenr)
  real(kind=dp), intent(in) :: zeta_u(lenr), psir_pp(lenr), psit_pp(lenr), psir_pu(lenr)
  real(kind=dp), intent(in) :: psit_pu(lenr), psir_uu(lenr), psit_uu(lenr), boxsize
  real(kind=dp), intent(out) :: cc(lenx)

  integer :: i

  do i = 1, lenx
    call get_cc_float(x1(i), x2(i), y1(i), y2(i), z1(i), z2(i), ex1(i), ex2(i) &
      , ey1(i), ey2(i), ez1(i), ez2(i), type1(i), type2(i), adot_phi, adot_vel &
      , logr, xi, zeta_p, zeta_u, psir_pp, psit_pp, psir_pu, psit_pu, psir_uu &
      , psit_uu, boxsize, lenr, cc(i))
  end do

end subroutine get_cc_arrays
