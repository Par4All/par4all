      PROGRAM ALIAS
      REAL A(10,10), B(500)
      REAL X(100,100)
      EQUIVALENCE (A(1,1),B(1))
      EQUIVALENCE (M,N)
      COMMON W
      CALL FOO1(M,N)
      CALL FOO1(W,K)
      CALL FOO1(K,L)
      CALL FOO2(A(1,1),B(1))
      DO I = 1,10
         DO J= 1,10
            CALL FOO2(X(I,J),X(2*I,2*J))
         ENDDO
      ENDDO
      END

      SUBROUTINE FOO1(V1,V2)
      COMMON W
      V1 = 1
      CALL FOO3(V1,X,V2)
      END
      
      SUBROUTINE FOO2(XV1,XV2)
      REAL XV1(100), XV2(100)
      DO I=1,10
         XV1(I) = XV2(I)
      ENDDO
      END
      
      SUBROUTINE FOO3(F1,F2,F3)
      F1 = 30
      END








