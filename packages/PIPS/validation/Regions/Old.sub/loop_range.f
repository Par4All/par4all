C     If the loop range expression is modifed by the loop body, 
C     the modified variable is projected from the region of the loop,
C     we have an exact region which is not correct. 

      PROGRAM LOOP_RANGE
      COMMON N
      REAL A(10)
      DO I = 1,N
         X = A(I)
         CALL SUB
      ENDDO
      END

      SUBROUTINE SUB
      COMMON N
      N = NGV
      END
