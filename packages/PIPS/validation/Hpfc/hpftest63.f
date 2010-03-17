c empty I/O statements 
      program hpftest63
      integer n, i
      parameter(n=10)
      integer a(n)
chpf$ template t(n)
chpf$ processors p(2)
chpf$ distribute t(block) onto p
chpf$ align a(i) with t(i)
      print *, 'hpftest63 running'
chpf$ independent(i)
      do i=1, n
         a(i)=n-i
      enddo
      print *, (a(i), i=n, n-1)
      do i=1, 4, 2
         print *, a(i)
      enddo
      do i=4, 2, 1
         print *, a(i)
      enddo
      print *, (a(i), i=1, n)
      end
