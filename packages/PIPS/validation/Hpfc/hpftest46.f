      program hpftest46
      parameter (n=100)
      real a(n,n), b(2*n-2,n)
chpf$ template t(2*n-2,n)
chpf$ processors p(2,2)
chpf$ align a(i,j) with t(i+98,j)
chpf$ align b(i,j) with t(i,j)
chpf$ distribute t(block,block) onto p
      j=0
chpf$ independent(j,i)
      do j=1, n
         do i=1, n
            a(i,j) = 2.0
         enddo
      enddo

chpf$ independent(j,i)
      do j=1, n
         do i=n-2, 2*n-2
            b(i,j) = a(i-n,j)
         enddo
      enddo

      end
