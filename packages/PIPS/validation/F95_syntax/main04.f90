! Previous bug : 
! allocate(a(100),stat=ierr) => ALLOCATE(A(100),STAT=, IERR)

program main

    implicit none
    integer ierr
    integer, allocatable :: a(:)
    allocate(a(100),stat=ierr)

end program main
