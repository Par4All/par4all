      program equiv02

C     Check that the loop is not parallelized because of aliasing
C     between T and U

      real t(10), u(10)
      common /c_u/u
      equivalence (t(1),u(1))

      do i = 2, 10
         t(i) = u(i-1)
      enddo

      end
