C Nga Nguyen 5 March 2002 : this example violates the Fortran standard (12.8.2.3)
C I7, J7, I8 cannot be in the input list, since they are do-variables of the implied DO
C or loop index. 

      PROGRAM LOOPS
      INTEGER B(20), A(20,20), L5, U5, S5

      DO I1 = 1,20
         B(I1) = 0
      ENDDO      


      DO I2 = 1,20,2
         B(I2) = 0
      ENDDO

      DO I3 = 20,1,-1
         B(I3) = 0
      ENDDO      

      DO I4 = 20,1,-2
         B(I4) = 0
      ENDDO  

      READ (4,*) L5
      READ (4,*) U5
      READ (4,*) S5

      DO I5 = L5,U5,S5
         B(I5) = 0
      ENDDO


      READ *, (B(J1), J1 = 1,20)
      READ *, (B(J2), J2 = 1,20,2)
      READ *, (B(J3), J3 = 20,1,-1)
      READ *, (B(J4), J4 = 20,1,-2)
      READ *, (B(J5), J5 = L5,U5,S5)

      READ *, ((A(I6,J6), J6 = 1, 20), I6 = 1,20)

      DO I7 = 1, 20
         READ *, (A(I7,J7), I7, J7, J7 = 1, 20)
      ENDDO

      DO I8 = 1, 20
         READ *, (A(I8,J8), J8, J8 = 1, 20)
      ENDDO

      END



