      program hpftest43
      real a(20), b(20), h
chpf$ template t(22)
chpf$ align a(i) with t(i)
chpf$ align b(i) with t(i+2)
chpf$ processors p(4)
chpf$ distribute t(block) onto p
      print *, 'enter h'
      read *, h
chpf$ independent(i)
      do i=1,20
         b(i) = i+h
      enddo
chpf$ independent(i)
      do i=3,18
         a(i) = b(i-2)+b(i-1)
      enddo
      do i=1,20
         print *, 'a(',i,') = ',a(i)
      enddo
      end
