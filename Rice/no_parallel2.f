      PROGRAM NO_PARALLEL2
      INTEGER I,J,K,L,M

      REAL*8 R(10,10,10,10)
      real*8 R2

      DO WHILE (R2.GT.0.0D0)
         DO K = 1, 10
            DO J = 1, 10
               DO I = 1, 10
                  DO L = 1, 10
                     R2 = R2+R(L,I,J,K)*R(L,I,J,K)
                  ENDDO
               ENDDO
            ENDDO
         ENDDO
      ENDDO

      DO K = 1, 10
         DO J = 1, 10
            DO I = 1, 10
               DO L = 1, 10
                  R2 = R2+R(L,I,J,K)*R(L,I,J,K)
               ENDDO
            ENDDO
         ENDDO
      ENDDO

      write(10,*) 'result =', R2

      END
