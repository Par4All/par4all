      subroutine data17

C     Use parameters in DATA statements

      parameter (m=10)

      integer tab(m)

      data tab,i/m*m,m/

      j = m

      print *, i, j

      end
