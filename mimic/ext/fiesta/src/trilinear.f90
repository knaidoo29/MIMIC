include "grid.f90"


subroutine trilinear_periodic(fgrid, x, y, z, xbox, ybox, zbox, ngridx, ngridy, &
  ngridz, npart, f)

  ! Trilinear interpolation of field defined on a grid.
  !
  ! Parameters
  ! ----------
  ! fgrid : array
  !   Field values on the grid.
  ! x : array
  !   X coordinates where we need interpolated values.
  ! y : array
  !   Y coordinates where we need interpolated values.
  ! z : array
  !   Z coordinates where we need interpolated values.
  ! xbox, ybox, zbox : float
  !   Size of the box.
  ! ngridx, ngridy, ngridz : int
  !   Size of the grid along each axis.
  ! npart : int
  !   Number of particles.
  !
  ! Returns
  ! -------
  ! f : array
  !   Interpolated field values.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  ! define variables

  integer, intent(in) :: ngridx, ngridy, ngridz, npart
  real(kind=dp), intent(in) :: fgrid(ngridx*ngridy*ngridz)
  real(kind=dp), intent(in) :: x(npart), y(npart), z(npart), xbox, ybox, zbox
  real(kind=dp), intent(out) :: f(npart)

  real(kind=dp) :: dx, dy, dz, xp, yp, zp, xg1, xg2, yg1, yg2, zg1, zg2, xd, yd, zd
  real(kind=dp) :: f111, f112, f121, f122, f211, f212, f221, f222
  real(kind=dp) :: f11, f12, f21, f22, f1, f2, minx
  integer :: q111, q112, q121, q122, q211, q212, q221, q222
  integer :: i, ix1, ix2, iy1, iy2, iz1, iz2

  minx = 0.

  dx = xbox / real(ngridx)
  dy = ybox / real(ngridy)
  dz = zbox / real(ngridz)

  do i = 1, npart

    xp = x(i)
    yp = y(i)
    zp = z(i)

    if (xp - dx/2. < 0.) then
      xp = xp + xbox
    end if
    if (yp - dy/2. < 0.) then
      yp = yp + ybox
    end if
    if (zp - dz/2. < 0.) then
      zp = zp + zbox
    end if

    ix1 = int((xp - dx/2.) / dx)
    call xgrid(ix1, dx, minx, xg1)

    ix2 = ix1 + 1
    call xgrid(ix2, dx, minx, xg2)

    if (ix2 .eq. ngridx) then
      ix2 = ix2 - ngridx
    end if

    iy1 = int((yp - dy/2.) / dy)
    call xgrid(iy1, dy, minx, yg1)

    iy2 = iy1 + 1
    call xgrid(iy2, dy, minx, yg2)

    if (iy2 .eq. ngridy) then
      iy2 = iy2 - ngridy
    end if

    iz1 = int((zp - dz/2.) / dz)
    call xgrid(iz1, dz, minx, zg1)

    iz2 = iz1 + 1
    call xgrid(iz2, dz, minx, zg2)

    if (iz2 .eq. ngridz) then
      iz2 = iz2 - ngridz
    end if

    ! surround points in the grid of a single point for interpolation.

    q111 = iz1 + ngridz*(iy1 + ngridy*ix1) + 1
    q112 = iz1 + ngridz*(iy1 + ngridy*ix2) + 1
    q121 = iz1 + ngridz*(iy2 + ngridy*ix1) + 1
    q122 = iz1 + ngridz*(iy2 + ngridy*ix2) + 1
    q211 = iz2 + ngridz*(iy1 + ngridy*ix1) + 1
    q212 = iz2 + ngridz*(iy1 + ngridy*ix2) + 1
    q221 = iz2 + ngridz*(iy2 + ngridy*ix1) + 1
    q222 = iz2 + ngridz*(iy2 + ngridy*ix2) + 1

    f111 = fgrid(q111)
    f112 = fgrid(q112)
    f121 = fgrid(q121)
    f122 = fgrid(q122)
    f211 = fgrid(q211)
    f212 = fgrid(q212)
    f221 = fgrid(q221)
    f222 = fgrid(q222)

    xd = (xp - xg1) / (xg2 - xg1)
    yd = (yp - yg1) / (yg2 - yg1)
    zd = (zp - zg1) / (zg2 - zg1)

    f11 = f111*(1-xd) + f112*xd
    f21 = f211*(1-xd) + f212*xd
    f12 = f121*(1-xd) + f122*xd
    f22 = f221*(1-xd) + f222*xd

    f1 = f11*(1-yd) + f12*yd
    f2 = f21*(1-yd) + f22*yd

    f(i) = f1*(1-zd) + f2*zd

  end do

end subroutine trilinear_periodic


subroutine trilinear_nonperiodic(fgrid, x, y, z, xbox, ybox, zbox, ngridx, &
  ngridy, ngridz, npart, f)

  ! Trilinear interpolation of field defined on a grid.
  !
  ! Parameters
  ! ----------
  ! fgrid : array
  !   Field values on the grid.
  ! x : array
  !   X coordinates where we need interpolated values.
  ! y : array
  !   Y coordinates where we need interpolated values.
  ! z : array
  !   Z coordinates where we need interpolated values.
  ! xbox, ybox, zbox : float
  !   Size of the box.
  ! ngridx, ngridy, ngridz : int
  !   Size of the grid along each axis.
  ! npart : int
  !   Number of particles.
  !
  ! Returns
  ! -------
  ! f : array
  !   Interpolated field values.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  ! define variables

  integer, intent(in) :: ngridx, ngridy, ngridz, npart
  real(kind=dp), intent(in) :: fgrid(ngridx*ngridy*ngridz)
  real(kind=dp), intent(in) :: x(npart), y(npart), z(npart), xbox, ybox, zbox
  real(kind=dp), intent(out) :: f(npart)

  real(kind=dp) :: dx, dy, dz, xp, yp, zp, xg1, xg2, yg1, yg2, zg1, zg2, xd, yd, zd
  real(kind=dp) :: f111, f112, f121, f122, f211, f212, f221, f222
  real(kind=dp) :: f11, f12, f21, f22, f1, f2, minx
  integer :: q111, q112, q121, q122, q211, q212, q221, q222
  integer :: i, ix1, ix2, iy1, iy2, iz1, iz2

  minx = 0.

  dx = xbox / real(ngridx)
  dy = ybox / real(ngridy)
  dz = zbox / real(ngridz)

  do i = 1, npart

    xp = x(i)
    yp = y(i)
    zp = z(i)

    if (xp - dx/2. < 0.) then
      ix1 = -1
      ix2 = 0
      call xgrid(ix1, dx, minx, xg1)
      call xgrid(ix2, dx, minx, xg2)
      ix1 = 0
    else if (xp > xbox - dx/2.) then
      ix1 = ngridx - 1
      ix2 = ngridx
      call xgrid(ix1, dx, minx, xg1)
      call xgrid(ix2, dx, minx, xg2)
      ix2 = ngridx - 1
    else
      ix1 = int((xp - dx/2.) / dx)
      ix2 = ix1 + 1
      call xgrid(ix1, dy, minx, xg1)
      call xgrid(ix2, dy, minx, xg2)
    end if

    if (yp - dy/2. < 0.) then
      iy1 = -1
      iy2 = 0
      call xgrid(iy1, dy, minx, yg1)
      call xgrid(iy2, dy, minx, yg2)
      iy1 = 0
    else if (yp > ybox - dy/2.) then
      iy1 = ngridy - 1
      iy2 = ngridy
      call xgrid(iy1, dy, minx, yg1)
      call xgrid(iy2, dy, minx, yg2)
      iy2 = ngridy - 1
    else
      iy1 = int((yp - dy/2.) / dy)
      iy2 = iy1 + 1
      call xgrid(iy1, dy, minx, yg1)
      call xgrid(iy2, dy, minx, yg2)
    end if

    if (zp - dz/2. < 0.) then
      iz1 = -1
      iz2 = 0
      call xgrid(iz1, dz, minx, zg1)
      call xgrid(iz2, dz, minx, zg2)
      iz1 = 0
    else if (zp > zbox - dz/2.) then
      iz1 = ngridz - 1
      iz2 = ngridz
      call xgrid(iz1, dz, minx, zg1)
      call xgrid(iz2, dz, minx, zg2)
      iz2 = ngridz - 1
    else
      iz1 = int((zp - dz/2.) / dz)
      iz2 = iz1 + 1
      call xgrid(iz1, dz, minx, zg1)
      call xgrid(iz2, dz, minx, zg2)
    end if

    ! surround points in the grid of a single point for interpolation.

    q111 = iz1 + ngridz*(iy1 + ngridy*ix1) + 1
    q112 = iz1 + ngridz*(iy1 + ngridy*ix2) + 1
    q121 = iz1 + ngridz*(iy2 + ngridy*ix1) + 1
    q122 = iz1 + ngridz*(iy2 + ngridy*ix2) + 1
    q211 = iz2 + ngridz*(iy1 + ngridy*ix1) + 1
    q212 = iz2 + ngridz*(iy1 + ngridy*ix2) + 1
    q221 = iz2 + ngridz*(iy2 + ngridy*ix1) + 1
    q222 = iz2 + ngridz*(iy2 + ngridy*ix2) + 1

    f111 = fgrid(q111)
    f112 = fgrid(q112)
    f121 = fgrid(q121)
    f122 = fgrid(q122)
    f211 = fgrid(q211)
    f212 = fgrid(q212)
    f221 = fgrid(q221)
    f222 = fgrid(q222)

    xd = (xp - xg1) / (xg2 - xg1)
    yd = (yp - yg1) / (yg2 - yg1)
    zd = (zp - zg1) / (zg2 - zg1)

    f11 = f111*(1-xd) + f112*xd
    f21 = f211*(1-xd) + f212*xd
    f12 = f121*(1-xd) + f122*xd
    f22 = f221*(1-xd) + f222*xd

    f1 = f11*(1-yd) + f12*yd
    f2 = f21*(1-yd) + f22*yd

    f(i) = f1*(1-zd) + f2*zd

  end do

end subroutine trilinear_nonperiodic


subroutine trilinear_axisperiodic(fgrid, x, y, z, xbox, ybox, zbox, perix, periy, &
  periz, ngridx, ngridy, ngridz, npart, f)

  ! Trilinear interpolation of field defined on a grid.
  !
  ! Parameters
  ! ----------
  ! fgrid : array
  !   Field values on the grid.
  ! x : array
  !   X coordinates where we need interpolated values.
  ! y : array
  !   Y coordinates where we need interpolated values.
  ! z : array
  !   Z coordinates where we need interpolated values.
  ! xbox, ybox, zbox : float
  !   Size of the box.
  ! perix, periy, periz : int
  !   0 = non-periodic, 1 = periodic
  ! ngridx, ngridy, ngridz : int
  !   Size of the grid along each axis.
  ! npart : int
  !   Number of particles.
  !
  ! Returns
  ! -------
  ! f : array
  !   Interpolated field values.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  ! define variables

  integer, intent(in) :: ngridx, ngridy, ngridz, npart, perix, periy, periz
  real(kind=dp), intent(in) :: fgrid(ngridx*ngridy*ngridz)
  real(kind=dp), intent(in) :: x(npart), y(npart), z(npart), xbox, ybox, zbox
  real(kind=dp), intent(out) :: f(npart)

  real(kind=dp) :: dx, dy, dz, xp, yp, zp, xg1, xg2, yg1, yg2, zg1, zg2, xd, yd, zd
  real(kind=dp) :: f111, f112, f121, f122, f211, f212, f221, f222
  real(kind=dp) :: f11, f12, f21, f22, f1, f2, minx
  integer :: q111, q112, q121, q122, q211, q212, q221, q222
  integer :: i, ix1, ix2, iy1, iy2, iz1, iz2

  minx = 0.

  dx = xbox / real(ngridx)
  dy = ybox / real(ngridy)
  dz = zbox / real(ngridz)

  do i = 1, npart

    xp = x(i)
    yp = y(i)
    zp = z(i)

    if (perix == 1) then
      if (xp - dx/2. < 0.) then
        xp = xp + xbox
      end if
      ix1 = int((xp - dx/2.) / dx)
      call xgrid(ix1, dx, minx, xg1)
      ix2 = ix1 + 1
      call xgrid(ix2, dx, minx, xg2)
      if (ix2 .eq. ngridx) then
        ix2 = ix2 - ngridx
      end if
    else
      if (xp - dx/2. < 0.) then
        ix1 = -1
        ix2 = 0
        call xgrid(ix1, dx, minx, xg1)
        call xgrid(ix2, dx, minx, xg2)
        ix1 = 0
      else if (xp > xbox - dx/2.) then
        ix1 = ngridx - 1
        ix2 = ngridx
        call xgrid(ix1, dx, minx, xg1)
        call xgrid(ix2, dx, minx, xg2)
        ix2 = ngridx - 1
      else
        ix1 = int((xp - dx/2.) / dx)
        ix2 = ix1 + 1
        call xgrid(ix1, dy, minx, xg1)
        call xgrid(ix2, dy, minx, xg2)
      end if
    end if

    if (periy == 1) then
      if (yp - dy/2. < 0.) then
        yp = yp + ybox
      end if
      iy1 = int((yp - dy/2.) / dy)
      call xgrid(iy1, dy, minx, yg1)

      iy2 = iy1 + 1
      call xgrid(iy2, dy, minx, yg2)

      if (iy2 .eq. ngridy) then
        iy2 = iy2 - ngridy
      end if
    else
      if (yp - dy/2. < 0.) then
        iy1 = -1
        iy2 = 0
        call xgrid(iy1, dy, minx, yg1)
        call xgrid(iy2, dy, minx, yg2)
        iy1 = 0
      else if (yp > ybox - dy/2.) then
        iy1 = ngridy - 1
        iy2 = ngridy
        call xgrid(iy1, dy, minx, yg1)
        call xgrid(iy2, dy, minx, yg2)
        iy2 = ngridy - 1
      else
        iy1 = int((yp - dy/2.) / dy)
        iy2 = iy1 + 1
        call xgrid(iy1, dy, minx, yg1)
        call xgrid(iy2, dy, minx, yg2)
      end if
    end if

    if (periz == 1) then
      if (zp - dz/2. < 0.) then
        zp = zp + zbox
      end if
      iz1 = int((zp - dz/2.) / dz)
      call xgrid(iz1, dz, minx, zg1)

      iz2 = iz1 + 1
      call xgrid(iz2, dz, minx, zg2)

      if (iz2 .eq. ngridz) then
        iz2 = iz2 - ngridz
      end if

    else
      if (zp - dz/2. < 0.) then
        iz1 = -1
        iz2 = 0
        call xgrid(iz1, dz, minx, zg1)
        call xgrid(iz2, dz, minx, zg2)
        iz1 = 0
      else if (zp > zbox - dz/2.) then
        iz1 = ngridz - 1
        iz2 = ngridz
        call xgrid(iz1, dz, minx, zg1)
        call xgrid(iz2, dz, minx, zg2)
        iz2 = ngridz - 1
      else
        iz1 = int((zp - dz/2.) / dz)
        iz2 = iz1 + 1
        call xgrid(iz1, dz, minx, zg1)
        call xgrid(iz2, dz, minx, zg2)
      end if
    end if

    ! surround points in the grid of a single point for interpolation.

    q111 = iz1 + ngridz*(iy1 + ngridy*ix1) + 1
    q112 = iz1 + ngridz*(iy1 + ngridy*ix2) + 1
    q121 = iz1 + ngridz*(iy2 + ngridy*ix1) + 1
    q122 = iz1 + ngridz*(iy2 + ngridy*ix2) + 1
    q211 = iz2 + ngridz*(iy1 + ngridy*ix1) + 1
    q212 = iz2 + ngridz*(iy1 + ngridy*ix2) + 1
    q221 = iz2 + ngridz*(iy2 + ngridy*ix1) + 1
    q222 = iz2 + ngridz*(iy2 + ngridy*ix2) + 1

    f111 = fgrid(q111)
    f112 = fgrid(q112)
    f121 = fgrid(q121)
    f122 = fgrid(q122)
    f211 = fgrid(q211)
    f212 = fgrid(q212)
    f221 = fgrid(q221)
    f222 = fgrid(q222)

    xd = (xp - xg1) / (xg2 - xg1)
    yd = (yp - yg1) / (yg2 - yg1)
    zd = (zp - zg1) / (zg2 - zg1)

    f11 = f111*(1-xd) + f112*xd
    f21 = f211*(1-xd) + f212*xd
    f12 = f121*(1-xd) + f122*xd
    f22 = f221*(1-xd) + f222*xd

    f1 = f11*(1-yd) + f12*yd
    f2 = f21*(1-yd) + f22*yd

    f(i) = f1*(1-zd) + f2*zd

  end do

end subroutine trilinear_axisperiodic
