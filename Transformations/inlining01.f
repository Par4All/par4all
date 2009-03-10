      SUBROUTINE FOo
      PRINT *, 'LIVE'
      END

      SUBROUTINE BAR
      CALL FOO()
      END
