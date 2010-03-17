      program equiv03

C     Check that aliased variable T is seen as in the common C_U
C     when a debugging option is used to print out common layouts

      real t(10), u(10)
      common /c_u/u
      equivalence (t(1),u(1))

      do i = 2, 10
         t(i) = u(i-1)
      enddo

      end
