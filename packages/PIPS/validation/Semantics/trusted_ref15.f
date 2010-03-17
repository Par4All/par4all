      subroutine trusted_ref15(m1)

C     Check that STOP statements are really used to generate information
C     for later transformers

      if(m1.lt.1) stop

      print *, m1

      m1 = m1 + 1

      end
