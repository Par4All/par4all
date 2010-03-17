c full copy test
      program fullcopy
      integer n, i, j
      parameter (n=512)
      real a(n,n), b(n,n)
cfcd$ setbool('HPFC_LAZY_MESSAGES',0)
cfcd$ setbool('HPFC_USE_BUFFERS', 0)
chpf$ dynamic a, b
chpf$ template tc(n,n), tb(n,n)
chpf$ processors p(2,2)
chpf$ align a(i,j) with tc(i,j)
chpf$ align b(i,j) with tb(i,j)
chpf$ distribute tc(cyclic,cyclic) onto p
chpf$ distribute tb(block,block) onto p
c
c initialize b
c
cfcd$ timeon
cfcd$ timeon
cfcd$ timeoff('empty measure')
chpf$ independent(j,i)
      do j=1, n
         do i=1, n
            b(i,j) = real(i+j)
         enddo
      enddo
      print *, (b(i,i+3), i=1, 10)
cfcd$ synchro
c
c realign b on tc => BLOCK to CYCLIC
c
chpf$ realign b(i,j) with tc(i,j)
c
c first a simple copy
c
chpf$ independent(j,i)
      do j=1, n
         do i=1, n
            a(i,j) = b(i,j)
         enddo
      enddo
c
c now transpose !
c
cfcd$ timeon
chpf$ realign a(i,j) with tc(j,i)
chpf$ independent(j,i)
      do j=1, n
         do i=1, n
            b(i,j) = a(j,i)
         enddo
      enddo
cfcd$ timeoff('transposition')
      print *, (a(i+3,i), i=1, 10)
cfcd$ timeoff('whole time')
      end
