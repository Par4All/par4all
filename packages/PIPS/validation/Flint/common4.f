        program common4
        common /bar/ x(3),y(3),j(3)
        call foo
        end
        subroutine foo
        common /bar/ y(2),x(4),i(2),j
        end
