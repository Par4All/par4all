! READ function

! read (unit=1,rec=1) a => READ (UNIT=1,FMT=*,REC=1) A
!
! Compilation error: FMT=* is not allowed with a REC= specifier at (1)


! Bug: the function return type replaced by "OVERLOADED"

program main
    
    implicit none
    integer :: a

    a = 0
    open(unit=1,file='main08.f90',form='unformatted', access='direct',status='old',recl=1000)

    read(1,rec=1) a

end program main
