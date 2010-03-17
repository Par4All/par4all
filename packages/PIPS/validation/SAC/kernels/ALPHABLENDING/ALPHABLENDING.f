      SUBROUTINE ALPHABLENDING(result, src, dest, alpha)

      INCLUDE 'ALPHABLENDING_INC.f'

      INTEGER i
      REAL result(1:SIZE), src(1:SIZE), dest(1:SIZE), alpha

c      INCLUDE 'ALPHABLENDING_INC2.f'

      DO 10 i = 1, SIZE
          result(i) = alpha * src(i) + ( 1 - alpha ) * dest(i)
10    ENDDO

      RETURN
      END

      SUBROUTINE ALPHABLENDINGREF(result, src, dest, alpha)

      INCLUDE 'ALPHABLENDING_INC.f'

      INTEGER i
      REAL result(1:SIZE), src(1:SIZE), dest(1:SIZE), alpha

c      INCLUDE 'ALPHABLENDING_INC2.f'

      DO 10 i = 1, SIZE
          result(i) = alpha * src(i) + ( 1 - alpha ) * dest(i)
10    ENDDO

      RETURN
      END

      INTEGER FUNCTION CLOCK()
      
      CLOCK = 0

      RETURN
      END

      PROGRAM MAIN

      INCLUDE 'ALPHABLENDING_INC.f'

      INTEGER i, j
      INTEGER bef, af, diff
      INTEGER befRef, afRef, diffRef
      INTEGER CLOCK
      REAL  a(1:SIZE), b(1:SIZE), c(1:SIZE)
      REAL  aRef(1:SIZE), bRef(1:SIZE), 
     & cRef(1:SIZE)

      LOGICAL success

      REAL alpha
      alpha = 0.5

      DO 10 i = 1, SIZE
          b(i) = i
          c(i) = i
          bRef(i) = i
          cRef(i) = i
10    ENDDO

      bef = CLOCK()
      DO 20 i = 1, 2000
          CALL ALPHABLENDING(a, b, c, alpha)
20    ENDDO
      af = CLOCK()

      befRef = CLOCK()
      DO 21 i = 1, 2000
          CALL ALPHABLENDINGREF(aRef, bRef, cRef, alpha)
21    ENDDO
      afRef = CLOCK()

      success = .TRUE.

      DO 22 i = 1, SIZE
c         IF((ABS(a(i) - aRef(i)) / aRef(i)) .GT. 0.000001) THEN
          IF(a(i) .NE. aRef(i)) THEN
             success = .FALSE.
         ENDIF
22    ENDDO

      PRINT *, 'a = ', a(1)
      PRINT *, 'aRef = ', aRef(1)
      PRINT *, 'a = ', a(2)
      PRINT *, 'aRef = ', aRef(2)

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
