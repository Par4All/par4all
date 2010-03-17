      program test1

      integer I
      real  X(100)

C     AND expression

      DO 100 I =1,N
         IF ((I.LT.1).AND.(I.LE.50)) THEN                     
         ELSE
            X(I) = 1            
         ENDIF 
 100  CONTINUE

C     OR expression

      DO 200 I =1,N
         IF ((1.LE.I).OR.(I.LE.50)) THEN
         ELSE
            X(I) = 1
         ENDIF 
 200  CONTINUE

C     NOT expression

      DO 300 I =2,10
         IF (.NOT.(I.GT.1)) THEN
         ELSE
            X(I) = 0                                                         
         ENDIF 
 300  CONTINUE

      end












