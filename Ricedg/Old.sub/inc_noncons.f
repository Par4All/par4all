c     This is a test example for the case of loop increment 
c     no constant 
      subroutine inc(A,m,n,l)
      integer m,n,l
      real*8 A(2*n)

      do i = m, n, l
         do j = 1, i+1
            A(2*j) = A(2*i+1)
         enddo
      enddo
      end


