      program hpftest30
      integer a(10), b(10)
      integer i
chpf$ template t(10)
chpf$ align a(i), b(i) with t(i)
chpf$ processors p(2)
chpf$ distribute t(block) onto p
      print *, 'hpftest30 running'
chpf$ independent(i)
      do i=1, 10
         a(i)=i
      enddo
      b(1)=100
chpf$ independent(i)
      do i=2, 10
         b(i)=a(i-1)
      enddo
      do i=1, 10
         print *, 'b(',i,')=',b(i)
      enddo
      print *, 'hpftest30 ended'
      end
