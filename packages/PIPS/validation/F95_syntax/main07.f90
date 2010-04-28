! Using functions
! Bug: the function return type replaced by "OVERLOADED"

program main

    implicit none
    integer :: a,b,c,add1,add2
    a = 10
    b = 20

    c = add1(a,b)
    c = add2(a,b)

end program main

function add1(a, b)
    integer a, b, add1
    add1 = a + b
    return
end function add1


integer function add2(a, b)
    integer a, b
    add2 = a + b
    return
end function add2
