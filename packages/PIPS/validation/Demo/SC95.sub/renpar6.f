
      SUBROUTINE RENPAR6(A,N,K,M)
      INTEGER N,K,M,A(N)
      DIMENSION WORK(100,100)
      K = M * M
      DO I = 1,N
         DO J = 1,N
            WORK(J,K) = J + K
         ENDDO

         CALL INC1(K)

         DO J = 1,N
            WORK(J,K) = J * J - K * K
            A(I) = A(I) + WORK(J,K) + WORK(J,K-1)
         ENDDO
      ENDDO
      END

      
      SUBROUTINE INC1(I)
      I = I + 1
      END


      SUBROUTINE RENPAR6_2(A,N,K,M)
      INTEGER N,K,M,A(N)
      DIMENSION WORK(100,100)
      K0 = M * M
      DO I = 1,N
         K = K0+I-1
         DO J = 1,N
            WORK(J,K) = J + K
         ENDDO

         CALL INC1(K)

         DO J = 1,N
            WORK(J,K) = J * J - K * K
            A(I) = A(I) + WORK(J,K) + WORK(J,K-1)
         ENDDO
      ENDDO
      END

      

