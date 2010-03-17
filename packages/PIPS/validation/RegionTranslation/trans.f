      program trans
      integer size
      parameter (size=10000)
      integer a(size)
      read *, n
      call foo(a, n)
!!      print *, (a(i),i=1, size)
      end

      subroutine foo(b, n)
      integer n, b(n,10,10),i,j,k
      call bla(b(1,1,1), n)
      end

      subroutine bla(c, n)
      integer n, c(n,1), i, j
      do i=1,n
         do j=1, 50
            c(i,j) = i+j
         enddo
      enddo
      end

      
