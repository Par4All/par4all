* Example from the C source of Andreas Podelski 
*
* In case : I = K -1, there are no bound violations, 
* but we have to set at least :
* setproperty SEMANTICS_FIX_POINT_OPERATOR "derivative"
* activate TRANSFORMERS_INTER_FULL
* #activate PRECONDITIONS_INTER_FULL
* so that PIPS can prove it
* 
* In case : I = K, there are bound violations

      SUBROUTINE INSERTIONSORT(A, N)
      REAL A(0:N-1), X
      INTEGER N
      
      INTEGER I,K   
      
      DO 100 K=1,N-1
         X = A(K)
         I = K
C         I = K-1
         DO WHILE ((I .GE. 0).AND.( A(I) .GT. X))
            A(I+1) = A(I)
            I = I-1
         ENDDO
         A(I+1)=X
 100  CONTINUE
                                      
      END

