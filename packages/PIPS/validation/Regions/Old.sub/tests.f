C     REGIONS OF TESTS, WITH VARIOUS CONDITIONS
C     AND ARRAY ACCESSES

      SUBROUTINE TESTS(N, R, J)
      
      INTEGER A(20), B(20), N, J
      REAL R

C     ONLY ONE BRANCH
      IF (N.GT.3) THEN
         A(N) = N - 1
      ENDIF

C     TWO BRANCHES WITH THE SAME ARRAY ELEMENTS ACCESSED
      IF (N.EQ.3) THEN
         A(N) = A(N) + N
      ELSE
         A(N) = A(N) - N
      ENDIF
      
C     A REAL IN THE CONDITION AND THE SAME ELTS ACCESSED
      IF (R.GT.0) THEN
         B(N) = A(N) / R
      ELSE
         B(N) = A(N) * R
      ENDIF
      
C     TWO DIFFERENT ARRAY ELEMENTS ACCESSED IN THE BRANCHES
      IF (N.GT.3) THEN
         A(N) = N - 1
      ELSE
         A(N+1) = N + 1
      ENDIF

C     THE ARRAY INDICES DON'T DEPEND ON THE TEST VARIABLE
      IF (N.GT.3) THEN
         A(M) = N - 1
      ELSE
         A(M) = N + 1
      ENDIF

C     THE TEST VARIABLE IS AN ARRAY ELEMENT, BUT IS NOT 
C     ACCESSED IN THE BRANCHES
      IF (B(N).EQ.N) THEN
         A(N) = A(N) + 1
      ENDIF

C     THE TEST VARIABLE IS AN ARRAY ELEMENT, AND IS 
C     ACCESSED IN THE TWO BRANCHES
      IF (B(N).EQ.N) THEN
         A(N) = B(N) + 1
      ELSE
         A(N) = B(N) - 1
      ENDIF

C     THE TEST VARIABLE IS AN ARRAY ELEMENT, AND IS 
C     ACCESSED IN ONLY ONE BRANCH
      IF (B(N).EQ.N) THEN
         A(N) = B(N) + 1
      ELSE
         A(N) = A(N) - 1
      ENDIF

      END


      
