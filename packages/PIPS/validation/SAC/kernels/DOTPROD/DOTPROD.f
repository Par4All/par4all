      SUBROUTINE DOTPROD(a, b, c)

      INCLUDE 'DOTPROD_INC.f'

      INTEGER i
      INTEGER*2 a(1:1), b(1:SIZE), c(1:SIZE)

      a(1) = 0
      DO 10 i = 1, SIZE
          a(1) = a(1) + b(i) * c(i)
10    ENDDO

      RETURN
      END

      SUBROUTINE DOTPRODREF(a, b, c)

      INCLUDE 'DOTPROD_INC.f'

      INTEGER i
      INTEGER*2 a(1:1), b(1:SIZE), c(1:SIZE)

      a(1) = 0
      DO 10 i = 1, SIZE
          a(1) = a(1) + b(i) * c(i)
10    ENDDO

      RETURN
      END

      INTEGER FUNCTION CLOCK()
      
      CLOCK = 0

      RETURN
      END

      PROGRAM MAIN

      INCLUDE 'DOTPROD_INC.f'

      INTEGER i, j
      INTEGER bef, af, diff
      INTEGER befRef, afRef, diffRef
      INTEGER CLOCK
      INTEGER*2  a(1:1), b(1:SIZE), c(1:SIZE)
      INTEGER*2  aRef(1:1), bRef(1:SIZE), 
     & cRef(1:SIZE)

      LOGICAL success

      DO 10 i = 1, SIZE
          b(i) = i
          c(i) = i
          bRef(i) = i
          cRef(i) = i
10    ENDDO

      bef = CLOCK()
      DO 20 i = 1, 100000
          CALL DOTPROD(a, b, c)
20    ENDDO
      af = CLOCK()

      befRef = CLOCK()
      DO 21 i = 1, 100000
          CALL DOTPRODREF(aRef, bRef, cRef)
21    ENDDO
      afRef = CLOCK()

      success = .TRUE.

c      IF((ABS(a(i) - aRef(i)) / aRef(i)) .GT. 0.000001) THEN
      IF(a(1) .NE. aRef(1)) THEN
          success = .FALSE.
      ENDIF

      PRINT *, 'a = ', a(1)
      PRINT *, 'aRef = ', aRef(1)

      IF(success .EQV. .TRUE.) THEN
         PRINT *, 'SUCCESS'
      ELSE
         PRINT *, 'ERROR'
      ENDIF

      diff = af - bef
      diffRef = afRef - befRef
      PRINT *, 'time: ', diff
      PRINT *, 'reference time: ', diffRef

      RETURN
      END
