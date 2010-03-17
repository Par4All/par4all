      PROGRAM ALIAS
      INTEGER I,A(5)
      DO I=1,5
         A(I)=2
      ENDDO
      WRITE(*,*) 'BEFORE:',A
      CALL SUB(A,A(1))
      WRITE(*,*) 'AFTER: ' ,A
      END
      SUBROUTINE SUB(X,Y)
      INTEGER I,X(5),Y
      DO I=1,5
         X(I)=Y*X(I)
      ENDDO
      RETURN 
      END
