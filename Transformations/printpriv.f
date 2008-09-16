      program printpriv
      real b(10,10), t
c
      do i=1,10
         do j=1,10
            b(i,j)=0.
         enddo
         do k=1,10
            t=k
            b(i,k)=t
         enddo
      enddo
      end
c
      subroutine foo
      real b(10,10)
      do i=1,10
         do k=1,10
            b(i,k)=0.
         enddo
      enddo
      return
      end
