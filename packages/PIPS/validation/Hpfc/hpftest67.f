c test shifts
      program hpftest67
      integer n
      parameter(n=25)
      integer i, j, a(n,n)
chpf$ processors p(3)
chpf$ template t(n)
chpf$ align a(i,*) with t(i)
chpf$ distribute t(block) onto p
      print *, 'HPFTEST67 IS RUNNING'
chpf$ independent(j,i)
      do j=1, n
         do i=1, n
            a(i,j)=n*(j-1)+i-1
         enddo
      enddo
chpf$ independent(j,i)
      do j=1, n-2
         do i=1, 3
            a(i, j)=a(i+n-5, j)
         enddo
      enddo
chpf$ independent(j,i)
      do j=n-10, n-5
         do i=4, n-2
            a(i, j) = a(i, j-10)
         enddo
      enddo
chpf$ independent(j,i)
      do j=8, 17
         do i=7, 11
            a(i, j) = a(i+8, j)
         enddo
      enddo
      print *, ((a(i,j), i=1, n), j=1, n)
      end
