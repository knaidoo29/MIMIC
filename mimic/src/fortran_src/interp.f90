
subroutine interp_lin_float(x, f, lenx, xval, fillval, fval)

  ! Interpolate a given function with linear interpolation with equal spacing.
  !
  ! Parameters
  ! ----------
  ! x : array
  !   Coordinate values.
  ! f : array
  !   Function values.
  ! lenx : int
  !   Length of x.
  ! xval : float
  !   Values of linear interpolation.
  ! fillval : float
  !   Filler values for regions beyond the boundaries.
  !
  ! Returns
  ! -------
  ! fval : float
  !   Interpolated value.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: lenx
  real(kind=dp), intent(in) :: x(lenx), f(lenx), xval, fillval
  real(kind=dp), intent(out) :: fval

  integer :: ind
  real(kind=dp) :: xmin, xmax, dx, x1, x2, f1, f2

  xmin = x(1)
  xmax = x(lenx)
  dx = (xmax-xmin) / (lenx-1)

  ind = floor((xval - xmin)/dx) + 1

  if ((ind .GE. 1) .AND. (ind .LE. lenx)) then
    f1 = f(ind)
    f2 = f(ind+1)
    x1 = x(ind)
    x2 = x(ind+1)
    fval = f1 + (f2-f1)*(xval-x1)/(x2-x1)
  else
    fval = fillval
  end if

end subroutine interp_lin_float


subroutine interp_lin_array(x, f, lenx, xarr, lenxarr, fillval, farr)

  ! Interpolate a given function with linear interpolation with equal spacing.
  !
  ! Parameters
  ! ----------
  ! x : array
  !   Coordinate values.
  ! f : array
  !   Function values.
  ! lenx : int
  !   Length of x.
  ! xarr : array
  !   Values of linear interpolation.
  ! lenxarr : int
  !   Length of xarr
  ! fillval : float
  !   Filler values for regions beyond the boundaries.
  !
  ! Returns
  ! -------
  ! farr : float
  !   Interpolated values.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: lenx, lenxarr
  real(kind=dp), intent(in) :: x(lenx), f(lenx), xarr(lenxarr), fillval
  real(kind=dp), intent(out) :: farr(lenxarr)

  integer :: i

  do i = 1, lenxarr
    call interp_lin_float(x, f, lenx, xarr(i), fillval, farr(i))
  end do

end subroutine interp_lin_array


subroutine interp_log_float(logx, f, lenlogx, logxval, fmin, fmax, fval)

  ! Interpolate a given function with linear interpolation with equal log spacing.
  !
  ! Parameters
  ! ----------
  ! logx : array
  !   Coordinate values.
  ! f : array
  !   Function values.
  ! lenlogx : int
  !   Length of logx.
  ! logxval : float
  !   Values of linear interpolation.
  ! fmin, fmax : float
  !   Filler values for regions below and above the boundaries.
  !
  ! Returns
  ! -------
  ! fval : float
  !   Interpolated value.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: lenlogx
  real(kind=dp), intent(in) :: logx(lenlogx), f(lenlogx), logxval, fmin, fmax
  real(kind=dp), intent(out) :: fval

  integer :: ind
  real(kind=dp) :: logxmin, logxmax, dlogx, logx1, logx2, f1, f2

  logxmin = logx(1)
  logxmax = logx(lenlogx)
  dlogx = (logxmax-logxmin) / (lenlogx-1)

  ind = floor((logxval - logxmin)/dlogx) + 1

  if ((ind .GE. 1) .AND. (ind .LE. lenlogx)) then
    f1 = f(ind)
    f2 = f(ind+1)
    logx1 = logx(ind)
    logx2 = logx(ind+1)
    fval = f1 + (f2-f1)*(logxval-logx1)/(logx2-logx1)
  else if (ind .LT. 1) then
    fval = fmin
  else
    fval = fmax
  end if

end subroutine interp_log_float


subroutine interp_log_array(logx, f, lenlogx, logxarr, lenlogxarr, fmin, fmax, farr)

  ! Interpolate a given function with linear interpolation with equal log spacing.
  !
  ! Parameters
  ! ----------
  ! logx : array
  !   Coordinate values.
  ! f : array
  !   Function values.
  ! lenlogx : int
  !   Length of logx.
  ! logxarr : array
  !   Values of linear interpolation.
  ! fmin, fmax : float
  !   Filler values for regions below and above the boundaries.
  !
  ! Returns
  ! -------
  ! farr : float
  !   Interpolated values.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: lenlogx, lenlogxarr
  real(kind=dp), intent(in) :: logx(lenlogx), f(lenlogx), logxarr(lenlogxarr), fmin, fmax
  real(kind=dp), intent(out) :: farr(lenlogxarr)

  integer :: i

  do i = 1, lenlogxarr
    call interp_log_float(logx, f, lenlogx, logxarr(i), fmin, fmax, farr(i))
  end do

end subroutine interp_log_array
