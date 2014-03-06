C     Fortran version
C     use-def chains with if/else
C     no dependence between if and else case have to be done
      PROGRAM IF01F
      
      INTEGER R, A
      
      IF (.TRUE.) THEN
        R = 1
      ELSE
        R = 0
      END IF

      A = R
      RETURN
      END
