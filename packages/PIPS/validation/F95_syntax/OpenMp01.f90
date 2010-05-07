program OpenMp01

  integer, parameter :: size = 100
  integer :: i, j, b (size)

  i = 0
  j = 0


  do i=1,size
     b(i) = 0
  end do

  do j=1,size
     do i=1,size
        b(j) = b(j) + i
     end do
  end do

  do i=1,size
     do j=1,size
        b(j) = b(j) + i
     end do
  end do

end program OpenMp01
