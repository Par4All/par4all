program array01

  integer, parameter :: size = 100
  integer :: i, j, b (size, size)

  i = 0
  j = 0

  do i=1,size
     do j=1,size
        b(i,j) = i
     end do
  end do

end program array01
