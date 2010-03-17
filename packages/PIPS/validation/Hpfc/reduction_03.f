! reduction on a vector
      program redonvect
      integer n
      parameter(n=10)
      real*8 A(n,n), S(n)
      integer i,j

!hpf$ processors P(2,2)
!hpf$ distribute A onto P

!hpf$ independent(j,i)
      do j=1, n
         do i=1, n
            A(i,j) = n-i/(j+1)
         enddo
      enddo

!hpf$ independent
      do i=1, n
         S(i) = 0.0
      enddo

!hpf$ independent(j,i), reduction(S)
      do j=1, n
         do i=1, n
            S(i) = S(i) + A(i,j)
         enddo
      enddo

      print *, (S(i),i=1,n)

      end
