      program user_error03

C     Check that PIPS survives the user error in the call graph because
C     B is called with only one argument and that PIPS can then compute the
C     effects for B.

      call B(x)

      end

      subroutine B(x, y)

      print *, x+y

      end
