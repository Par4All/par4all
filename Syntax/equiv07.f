      program equiv07

C     Check that aliased variable T has proper offset

      real t(10), u(10)
      common /c_u/a, u
      equivalence (t(1),u(1))
      double precision a(10)

      do i = 2, 10
         t(i) = u(i-1)
      enddo

      end
