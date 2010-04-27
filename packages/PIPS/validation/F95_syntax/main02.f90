! Variable declaration and parameter

program main
    implicit none
    integer :: a,b
    a = 10
    b = 20
    call add(a,b)
end program main


subroutine add(a,b)
    implicit none
    integer :: a, b, c
    c = a+b
    print*,c
end subroutine
