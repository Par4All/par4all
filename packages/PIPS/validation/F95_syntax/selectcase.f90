! Handling simple select in Fortran95
PROGRAM main

    integer :: a, b
    a = 99;

    SELECT CASE (a)
        CASE (1,2,3,4)
            b = 1
        CASE (5)
            b = 2
        case default
            b = 3
    END SELECT

END PROGRAM main
