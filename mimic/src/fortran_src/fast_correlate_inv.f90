include "fast_correlate.f90"


subroutine corr1_dot_inv_dot_corr2(x1, x2, y1, y2, z1, z2, exi, eyi, ezi &
  , xc, yc, zc, exc, eyc, ezc, type1, type2, typec, adot_phi, adot_vel, logr, xi &
  , zeta_p, zeta_u, psir_pp, psit_pp , psir_pu, psit_pu, psir_uu, psit_uu, boxsize &
  , lenr, lenxi, lenxc, inv, field, mpi_rank, lenpro, lenpre, prefix)

  ! Computes correlation with point 1 dot inverse covariance dot the correlation
  ! with point 2.
  !
  ! Parameters
  ! ----------
  ! x1, y1, z1 : array
  !   Coordinates to compute correlation from point 1.
  ! x2, y2, z2 : array
  !   Coordinates to compute correlation from point 2.
  ! xc, yc, zc : array
  !   Coordinates of constraints.
  ! ex1, ey1, ez1 : float
  !   Vector for constraint outputs.
  ! ex2, ey2, ez2 : float
  !   Vector for constraint outputs.
  ! exc, eyc, ezc : array
  !   Vector for constraint inputs.
  ! type1 : int
  !   Constraint type for point 1.
  ! type2 : int
  !   Constraint type for point 2.
  ! typec : array
  !   Constraint type for inputs.
  ! adot_phi, adot_vel : float
  !   adot for displacement and velocity.
  ! logr : array
  !   Log10 of the radii.
  ! xi : array
  !   Density correlation values as a function of r.
  ! zeta_p, zeta_u : array
  !   Density-velocity correlation values as a function of r.
  ! psir_pp, psit_pp : array
  !   Displacment radial and tangential correlations values as a function of r.
  ! psir_pu, psit_pu : array
  !   Displacement-velocity radial and tangential correlations values as a function of r.
  ! psir_uu, psit_uu : array
  !   Velocity radial and tangential correlations values as a function of r.
  ! boxsize : float
  !   Boxsize.
  ! lenr : int
  !   Length of logr array and correlation functions.
  ! lenxi : int
  !   Length of 1 and 2 point arrays.
  ! lenxc : int
  !   Length of constraints.
  ! inv : array
  !   Inverse covariance, flattened to 1D array.
  ! field : array
  !   Output values.
  ! mpi_rank : int
  !   Rank of MPI object.
  ! lenpro : int
  !   Length of fortran progress bar.
  ! lenpre : int
  !   Length of fortran progress bar prefix.
  ! prefix : str
  !   Prefix string for fortran progress bar.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: lenr, lenxi, lenxc
  integer, intent(in) :: type1, type2, typec(lenxc)
  real(kind=dp), intent(in) :: x1(lenxi), x2(lenxi), y1(lenxi), y2(lenxi), z1(lenxi), z2(lenxi)
  real(kind=dp), intent(in) :: xc(lenxc), yc(lenxc), zc(lenxc)
  real(kind=dp), intent(in) :: exi, exc(lenxc), eyi, eyc(lenxc), ezi, ezc(lenxc), inv(lenxc*lenxc)
  real(kind=dp), intent(in) :: adot_phi, adot_vel, logr(lenr), xi(lenr), zeta_u(lenr), zeta_p(lenr)
  real(kind=dp), intent(in) :: psir_pp(lenr), psit_pp(lenr), psir_pu(lenr), psit_pu(lenr)
  real(kind=dp), intent(in) :: psir_uu(lenr), psit_uu(lenr), boxsize
  real(kind=dp), intent(out) :: field(lenxi)

  integer, intent(in) :: mpi_rank, lenpro, lenpre
  character, intent(in) :: prefix(lenpre)

  integer :: i, j, k, jj
  real(kind=dp) :: cc1(lenxc), cc2(lenxc), field_val

  do i=1, lenxi
    call get_cc_array2(x1(i), xc, y1(i), yc, z1(i), zc, exi, exc, eyi, eyc, ezi, ezc &
      , type1, typec, adot_phi, adot_vel, logr, xi, zeta_p, zeta_u, psir_pp, psit_pp &
      , psir_pu, psit_pu, psir_uu, psit_uu, boxsize, lenr, lenxc, cc1)
    call get_cc_array1(xc, x2(i), yc, y2(i), zc, z2(i), exc, exi, eyc, eyi, ezc, ezi &
      , typec, type2, adot_phi, adot_vel, logr, xi, zeta_p, zeta_u, psir_pp, psit_pp &
      , psir_pu, psit_pu, psir_uu, psit_uu, boxsize, lenr, lenxc, cc2)
    field_val = 0.
    do j=1, lenxc
      do k=1, lenxc
        jj = (j-1)*lenxc + (k-1) + 1
        field_val = field_val + cc1(j)*inv(jj)*cc2(k)
      end do
    end do
    field(i) = field_val
    if (mpi_rank .eq. 0) then
      call progress_bar(i, lenxi, lenpro, prefix, lenpre)
    end if
  end do

end subroutine corr1_dot_inv_dot_corr2



subroutine corr1_dot_inv_dot_corr2_array(x1, x2, y1, y2, z1, z2, exi, eyi, ezi &
  , xc, yc, zc, exc, eyc, ezc, type1, type2, typec, adot_phi, adot_vel, logr, xi &
  , zeta_p, zeta_u, psir_pp, psit_pp , psir_pu, psit_pu, psir_uu, psit_uu, boxsize &
  , lenr, lenxi, lenxc, inv, field, mpi_rank, lenpro, lenpre, prefix)

  ! Computes correlation with point 1 dot inverse covariance dot the correlation
  ! with point 2.
  !
  ! Parameters
  ! ----------
  ! x1, y1, z1 : array
  !   Coordinates to compute correlation from point 1.
  ! x2, y2, z2 : array
  !   Coordinates to compute correlation from point 2.
  ! xc, yc, zc : array
  !   Coordinates of constraints.
  ! ex1, ey1, ez1 : array
  !   Vector for constraint outputs.
  ! ex2, ey2, ez2 : array
  !   Vector for constraint outputs.
  ! exc, eyc, ezc : array
  !   Vector for constraint inputs.
  ! type1 : int
  !   Constraint type for point 1.
  ! type2 : int
  !   Constraint type for point 2.
  ! typec : array
  !   Constraint type for inputs.
  ! adot_phi, adot_vel : float
  !   adot for displacement and velocity.
  ! logr : array
  !   Log10 of the radii.
  ! xi : array
  !   Density correlation values as a function of r.
  ! zeta_p, zeta_u : array
  !   Density-velocity correlation values as a function of r.
  ! psir_pp, psit_pp : array
  !   Displacment radial and tangential correlations values as a function of r.
  ! psir_pu, psit_pu : array
  !   Displacement-velocity radial and tangential correlations values as a function of r.
  ! psir_uu, psit_uu : array
  !   Velocity radial and tangential correlations values as a function of r.
  ! boxsize : float
  !   Boxsize.
  ! lenr : int
  !   Length of logr array and correlation functions.
  ! lenxi : int
  !   Length of 1 and 2 point arrays.
  ! lenxc : int
  !   Length of constraints.
  ! inv : array
  !   Inverse covariance, flattened to 1D array.
  ! field : array
  !   Output values.
  ! mpi_rank : int
  !   Rank of MPI object.
  ! lenpro : int
  !   Length of fortran progress bar.
  ! lenpre : int
  !   Length of fortran progress bar prefix.
  ! prefix : str
  !   Prefix string for fortran progress bar.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: lenr, lenxi, lenxc
  integer, intent(in) :: type1, type2, typec(lenxc)
  real(kind=dp), intent(in) :: x1(lenxi), x2(lenxi), y1(lenxi), y2(lenxi), z1(lenxi), z2(lenxi)
  real(kind=dp), intent(in) :: xc(lenxc), yc(lenxc), zc(lenxc)
  real(kind=dp), intent(in) :: exi(lenxi), exc(lenxc), eyi(lenxi), eyc(lenxc)
  real(kind=dp), intent(in) :: ezi(lenxi), ezc(lenxc), inv(lenxc*lenxc)
  real(kind=dp), intent(in) :: adot_phi, adot_vel, logr(lenr), xi(lenr), zeta_u(lenr), zeta_p(lenr)
  real(kind=dp), intent(in) :: psir_pp(lenr), psit_pp(lenr), psir_pu(lenr), psit_pu(lenr)
  real(kind=dp), intent(in) :: psir_uu(lenr), psit_uu(lenr), boxsize
  real(kind=dp), intent(out) :: field(lenxi)

  integer, intent(in) :: mpi_rank, lenpro, lenpre
  character, intent(in) :: prefix(lenpre)

  integer :: i, j, k, jj
  real(kind=dp) :: cc1(lenxc), cc2(lenxc), field_val

  do i=1, lenxi
    call get_cc_array2(x1(i), xc, y1(i), yc, z1(i), zc, exi(i), exc, eyi(i), eyc, ezi(i), ezc &
      , type1, typec, adot_phi, adot_vel, logr, xi, zeta_p, zeta_u, psir_pp, psit_pp &
      , psir_pu, psit_pu, psir_uu, psit_uu, boxsize, lenr, lenxc, cc1)
    call get_cc_array1(xc, x2(i), yc, y2(i), zc, z2(i), exc, exi(i), eyc, eyi(i), ezc, ezi(i) &
      , typec, type2, adot_phi, adot_vel, logr, xi, zeta_p, zeta_u, psir_pp, psit_pp &
      , psir_pu, psit_pu, psir_uu, psit_uu, boxsize, lenr, lenxc, cc2)
    field_val = 0.
    do j=1, lenxc
      do k=1, lenxc
        jj = (j-1)*lenxc + (k-1) + 1
        field_val = field_val + cc1(j)*inv(jj)*cc2(k)
      end do
    end do
    field(i) = field_val
    if (mpi_rank .eq. 0) then
      call progress_bar(i, lenxi, lenpro, prefix, lenpre)
    end if
  end do

end subroutine corr1_dot_inv_dot_corr2_array
