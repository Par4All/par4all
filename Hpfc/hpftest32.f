      program hpftest32
      real a(20), b(20)
      integer i
chpf$ template t(20)
chpf$ align a(i), b(i) with t(i)
chpf$ processors p(4)
chpf$ distribute t(block) onto p
      print *, 'hpftest32 running'
chpf$ independent(i)
      do i=1, 20
         a(i)=i
      enddo
      b(1) = 100
chpf$ independent(i)
      do i=2, 20
            b(i)=a(i-1)
      enddo
      do i=1, 20
         print *, 'b(',i,')=',b(i)
      enddo
      print *, 'hpftest32 ended'
      end
