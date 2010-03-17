c test with common and subroutines
      program hpftest56
      print *, 'running'
      call initdata
      call computedata
      call printdata
      end
c
      subroutine initdata
      integer i
c here are the common declarations
      integer n, m
      parameter (n=20)
      parameter (m=4)
      common /data/ a(n,2), b
      integer a, b
chpf$ template t(n)
chpf$ align a(i,*) with t(i)
chpf$ processors p(m)
chpf$ distribute t(block) onto p
c end of the common declarations
      print *, 'initializing data'
chpf$ independent(i)
      do i=1, n
         a(i,1) = i
      enddo
chpf$ independent(i)
      do i=1, n
         a(i,2) = -1
      enddo
      end
c
      subroutine computedata
      integer i
c here are the common declarations
      integer n, m
      parameter (n=20)
      parameter (m=4)
      common /data/ a(n,2), b
      integer a, b
chpf$ template t(n)
chpf$ align a(i,*) with t(i)
chpf$ processors p(m)
chpf$ distribute t(block) onto p
c end of the common declarations
      print *, 'computing data'
chpf$ independent(i)
      do i=2, n-1
         a(i,2) = a(i-1,1)+a(i,1)+a(i+1,1)-2*i
      enddo
      end
c
      subroutine printdata
      integer i
c here are the common declarations
      integer n, m
      parameter (n=20)
      parameter (m=4)
      common /data/ a(n,2), b
      integer a, b
chpf$ template t(n)
chpf$ align a(i,*) with t(i)
chpf$ processors p(m)
chpf$ distribute t(block) onto p
c end of the common declarations
      print *, 'printing data'
      do i=1, n
         print *, i, a(i,2)
      enddo
      end
