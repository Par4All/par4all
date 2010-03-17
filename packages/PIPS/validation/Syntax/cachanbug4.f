      subroutine cachanbug4

C     Bug 4 found by Nicky Williams-Preston: core dump on hexadecimal
C     constants

      DATA I7F7F/Z'7F7F'/

      print *, i7f7f

      end

      subroutine cachanbug4b

C     Bug 4 found by Nicky Williams-Preston: core dump on hexadecimal
C     constants

C     Check that a variation is properly detected too

      integer z

      parameter (z=10)

      DATA I7F7F/Z'7F7F'/
c      DATA I7F7F/Z/

      print *, i7f7f

      end

      subroutine cachanbug4c

C     Bug 4 found by Nicky Williams-Preston: core dump on hexadecimal
C     constants

C     Check that PIPS is OK for the normal case: compute preconditions

      parameter (m=10)

      integer tab(m)

      data tab,i/m*m,m/

      j = m

      print *, i, j

      end

