! Assertion failed: there is no else if bloc

program main

    integer a

    a = 0

! Simple test
    if  (a == 0) then
        a = 1
    endif

! Test with else
    if  (a == 0) then
        a = 1
    else
        a = 3
    endif

! This one is with else if
    if (a == 0) then
        a = 1
    else if (a == 1) then
        a = 2
    else
        a = 3
    end if


end program main
