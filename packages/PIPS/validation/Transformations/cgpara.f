C     several small regression tests for the coarse grain 
C     parallelization phase of PIPS
      subroutine test1(A,n)
      integer n,A(n),i

      do i = 1,n
         a(i) = 0
      enddo
      end
         
      subroutine test2(A,n)
      integer n,A(n),i

      do i = 1,n
         a(i) = a(i-1)
      enddo
      end
         
      subroutine test3(A,n)
      integer n,A(n,n),i,j

      do i = 1,n
         do j =1,n
            a(i,j) = a(i+1,j)
         enddo
      enddo
      end
         
      subroutine test4(A,n,k)
      integer n,A(n),i,k

      do i = 1,n
         if (i.ne.k) then
            a(i) = 0
         endif
      enddo

      do i = 1, n
         if (i.le.k) then
            a(i) = 1
            goto 10
         endif

         if (i.ge.k) then
            a(i) = 2
            goto 10
         endif
 10      a(i) = a(i) + k
      enddo
      end
         
