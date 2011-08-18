      program ishftc01

      real I
      real J
      real k
      real H

      I = 2
      J = 3
      K = 3
!     gfortran insists on the arguments being integers
      H = ISHFTC(L1,L2,L3)
      PRINT *, 'H = ',H

      END
