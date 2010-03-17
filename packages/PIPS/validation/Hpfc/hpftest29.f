      program hpftest29
      real a(10), b(10), d
      integer i
chpf$ template t(10)
chpf$ align a(i), b(i) with t(i)
chpf$ processors p(2)
chpf$ distribute t(block) onto p
      print *, 'hpftest29 running'
      print *, 'enter real d'
      read *, d
chpf$ independent(i)
      do i=1, 10
         a(i)=i
      enddo
chpf$ independent(i)
      do i=1, 10
         b(i)=a(i)+d
      enddo
      do i=1, 10
         print *, 'b(',i,')=',b(i)
      enddo
      print *, 'hpftest29 ended'
      end
