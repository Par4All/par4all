C     different calls from different callers

      PROGRAM ALIAS
      COMMON /TOTO/ W(50)
      CALL FOO1(W(10))
      CALL FOO2(A,B,C)
      END
    
      SUBROUTINE FOO1(X1)
      COMMON /TOTO/ V(50)
      COMMON /TITI/ U(50)
      EQUIVALENCE (M,N)
      REAL X1(40)
      CALL FOO2(X1(10),M,N)
      CALL FOO2(U,K,L)
      CALL FOO2(T,K,K)
      END

      SUBROUTINE FOO2(Y1,Y2,Y3)
      REAL Y1(20)
      COMMON /TOTO/ W(50)
      DO I =1,20
         Y1(I) =I
      ENDDO
      CALL FOO3(Y1)
      END
      
      SUBROUTINE FOO3(Z1)
      COMMON /TOTO/ V(50)
      COMMON /TITI/ U(50)
      END













