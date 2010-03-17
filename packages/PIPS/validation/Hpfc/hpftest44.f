      program hpftest44
      parameter (n=20)
      real a(n), b(n), h
chpf$ template t(n+2)
chpf$ align a(i) with t(i)
chpf$ align b(i) with t(i+2)
chpf$ processors p(4)
chpf$ distribute t(block) onto p
      print *, 'enter h'
      read *, h
chpf$ independent(i)
      do i=1,n
         b(i) = i+h
      enddo
chpf$ independent(i)
      do i=3,n-2
         a(i) = b(i-2)+b(i-1)
      enddo
      do i=1,n
         print *, 'a(',i,') = ',a(i)
      enddo
      end
