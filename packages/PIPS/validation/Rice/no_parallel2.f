      PROGRAM NO_PARALLEL2
      INTEGER K

      real*8 R2

      R2 = 1500.0

      DO WHILE (R2.GT.0.0D0)
         DO K = 1, 10
            R2 = R2-5.0
         ENDDO
      ENDDO

      DO K = 1, 10
         R2 = R2+5.0
      ENDDO

      write(10,*) 'result =', R2

      END
