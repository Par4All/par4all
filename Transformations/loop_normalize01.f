      subroutine loop_normalize(A,n,i)
      real A(n)

      do j = 1, n, i
         A(i) = 0.
      enddo

      end
