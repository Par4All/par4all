PROGRAM main

    integer :: a, b

    SELECT CASE (a)
        CASE (1,2,3,4)
            b = 1
        CASE (5)
            b = 1
    END SELECT

END PROGRAM main
