! Check how pips handle select with different kind of range in case
! Note also that the default isn't the last case
program main

    integer :: a, b
    a = 99;
    select case (a)
        case (:-10, 10:) 
            b = 1
        case (-5:-3, 6:9)
            b = 2
        case default
            b = 6
        case (-2:2)
            b = 3
        case (3, 5)
            b = 4
        case (4)
            b = 5
    end select

end program main

