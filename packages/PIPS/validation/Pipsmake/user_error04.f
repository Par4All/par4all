      program user_error04

C     Check that PIPS survives a user error in a concurrent apply

      call B(x)

      end

      subroutine B(x, y)

      print *, x+y

      end
