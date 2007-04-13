      PROGRAM IFUNC01
      CALL BLA()
      END

      SUBROUTINE BLA()
      INTEGER FOO
      I = FOO()
      END

! fonctionne avec
!      INTEGER FUNCTION FOO()
!
! fonctionne si on renomme FOO en BLA et BLA en FOO
!
      FUNCTION FOO()
      INTEGER FOO
      FOO = 0
      END

