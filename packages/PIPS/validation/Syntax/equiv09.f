      program equiv09

C     Check that addresses for dynamic variables, implicitly or 
C     explicitly declared are correct

      real y(100)
      equivalence (x, z), (u, v)
      real u(100)

      do i = 2, 10
         y(i) = 0.
      enddo

      print *, i, j

      end
