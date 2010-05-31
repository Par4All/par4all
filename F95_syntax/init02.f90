program init01

integer, dimension (:), allocatable :: array
allocate (array(10), stat=ierr)
array(:) = 0

end program init01
