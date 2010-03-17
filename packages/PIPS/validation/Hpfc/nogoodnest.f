      program nogoodnest

      integer n
      parameter (n=100)

      real U(n,n), Uh(n,n)
      integer i,j

!hpf$ processors P(2)
!hpf$ distribute U(block,*) onto P
!hpf$ align with U:: Uh

!hpf$ independent
      do j=2, n-1
         Uh(1,j) = U(1, j)
!hpf$    independent
         do i=2, n-1
            Uh(i,j) = U(i,j) + U(i,j+1) + U(i,j-1)
         enddo
         Uh(n,j) = U(n,j)
      enddo

!hpf$ independent
      do j=2, n-1
!fcd$    local
         Uh(1,j) = U(1, j)
!fcd$    end local
!hpf$    independent
         do i=2, n-1
            Uh(i,j) = U(i,j) + U(i,j+1) + U(i,j-1)
         enddo
!fcd$    local
         Uh(n,j) = U(n,j)
!fcd$    end local
      enddo

      do j=2, n-1
         Uh(1,j) = U(1, j)
!hpf$    independent
         do i=2, n-1
            Uh(i,j) = U(i,j) + U(i,j+1) + U(i,j-1)
         enddo
         Uh(n,j) = U(n,j)
      enddo

      end
