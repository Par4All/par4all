c dynamic test
      program test68
      integer n, m, i, j
      parameter(n=20,m=30)
      real a(n,m)
cfcd$ setbool('HPFC_LAZY_MESSAGES',1)
cfcd$ setbool('HPFC_USE_BUFFERS', 1)
chpf$ template t(m,m)
chpf$ processors p(2,2)
chpf$ dynamic a, t
chpf$ align a(i,j) with t(i,j)
chpf$ distribute t(block,block) onto p
      print *, 'DYNAMIC TEST (68) RUNNING'
chpf$ independent(i,j)
      do j=1, n
         do i=1, n
            a(i, j) = real(i+j)/real(j-i+m+1)
         enddo
      enddo
      print *, a(10, 20)
c transpose
chpf$ realign a(i,j) with t(j,i)
      print *, 'after transpose'
chpf$ realign a(i,j) with t(m-i,j)
      print *, 'after unused remapping'
chpf$ realign a(i,j) with t(j,i)
      print *, a(10, 20)
      print *, (a(i+1,i), i=1, n-1)
chpf$ realign a(i,j) with t(i,j)
      print *, a(10, 20)
      end
