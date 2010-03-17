C     interprocedural propagation of regions concerning global 
C     arrays
      PROGRAM COMM
      COMMON /CA/ AA(5), AB(5), AC(5)
      COMMON /CB/ BA(5,5)
      COMMON /CC/ CA(5), CB(5,4)

      CALL COMM1()
      CALL COMM1B()
      CALL COMM2()
      CALL COMM2B()
      CALL COMM3()
      CALL COMM3B()

      print *, (AA(I), I = 1,5)
      print *, (AB(I), I = 1,5)
      print *, (AC(I), I = 1,5)
      print *, ((BA(I,J), I = 1,5) , J = 1,5)
      print *, (CA(I), I = 1,5)
      print *, ((CB(I,J), I = 1,5) , J = 1,4)
      END

      SUBROUTINE COMM1()
      COMMON /CA/ AD(7), AE(6), AF(2)      
      DO I = 1, 7
         AD(I) = I
      ENDDO
      DO I = 1, 6
         AE(I) = I
      ENDDO
      DO I = 1, 2
         AF(I) = I
      ENDDO
      END

      SUBROUTINE COMM1B()
      COMMON /CA/ AD(7), AE(6), AF(2)      
      DO I = 2, 6
         AD(I) = I
      ENDDO
      DO I = 2, 6
         AE(I) = I
      ENDDO
      DO I = 1, 2
         AF(I) = I
      ENDDO
      END

      SUBROUTINE COMM2()
      COMMON /CB/ BB(5),BC(4,5)
      DO I = 1, 5
         BB(I) = I
      ENDDO
      DO I = 1,4
         DO J = 1,5
            BC(I,J) = I+J
         ENDDO
      ENDDO
      END

      SUBROUTINE COMM2B()
      COMMON /CB/ BD(5),BE(5,4)
      DO I = 1, 5
         BD(I) = I
      ENDDO
      DO I = 1,5
         DO J = 1,4
            BE(I,J) = I+J
         ENDDO
      ENDDO
      END

      SUBROUTINE COMM3()
      COMMON /CC/ CD(5,5)

      DO I = 1,5
         DO J = 1,4
            CD(I,J) = I+J
         ENDDO
      ENDDO
      END

      SUBROUTINE COMM3B()
      COMMON /CC/ CE(5,5)

      DO I = 1,5
         DO J = 2,4
            CE(I,J) = I+J
         ENDDO
      ENDDO
      END
