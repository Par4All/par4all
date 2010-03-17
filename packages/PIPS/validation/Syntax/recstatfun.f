      SUBROUTINE STATFUN
      REAL FXI(4)
      F2(T) = T+T
      F3(T) = F2(-T)
      FXI(1) = F3(1)
      FXI(2) = F3(1+1)
      END
