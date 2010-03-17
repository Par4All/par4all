        program common5
        common /bar/ x(2),i,y(3),j(3)
        call foo
        end
        subroutine foo
        common /bar/ y(3),x(4),i(2)
        end
