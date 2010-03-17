c non perfectly nested parallel loop
      program hpftest65
      integer i, j
      integer n
      parameter (n=100)
      real a(n), b(n,n), tmp
chpf$ processors p(2,2)
chpf$ template t(n,n)
chpf$ align b(i,j) with t(i,j)
chpf$ distribute t(block,block) onto p
      print *, 'hpftest65 running'
      do i=1, n
         a(i)=111.0/i
      enddo
chpf$ independent(i)
      do i=1, n
         tmp = a(i)+7.5*i
chpf$ independent(j)
         do j=1, n
            b(j,i) = tmp + 17.3*j
         enddo
      enddo
      print *, ((b(i,j), i=45, 55), j=45, 55)
      end
