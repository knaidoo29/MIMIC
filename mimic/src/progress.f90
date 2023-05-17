

subroutine progress_bar(i, total, lenchar, prefix, lenpre)

  implicit none

  integer, intent(in) :: i, total, lenchar, lenpre
  real :: dx
  character :: prefix(lenpre), output(lenchar+lenpre)
  logical :: doprint
  integer :: j

  dx = real(total)/real(lenchar-2)

  doprint = .false.
  if (i .EQ. 1) then
    doprint = .true.
  else if (i .EQ. total) then
    doprint = .true.
  else if (floor((i-1)/dx) .NE. floor((i)/dx)) then
    doprint = .true.
  end if

  do j=1, lenpre
    output(j) = prefix(j)
  end do

  if (doprint .EQV. .true.) then
    output(lenpre+1) = "|"
    do j = 1, lenchar-2
      if (floor(i/dx) .LT. j) then
        output(lenpre+j+1) = "-"
      else
        output(lenpre+j+1) = "#"
      end if
    end do
    output(lenpre+lenchar) = "|"
    print *, output
  end if

end subroutine progress_bar
