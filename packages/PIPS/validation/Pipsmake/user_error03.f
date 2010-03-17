      program user_error03

C     Check that PIPS survives the user error in the call graph because
C     B is called with only one argument and that PIPS can parallelize B

      call B(x)

      end

      subroutine B(x, y)
      real u(10)

      do i = 1, 10
         u(i) = 0.
      enddo

      print *, x+y

      end
