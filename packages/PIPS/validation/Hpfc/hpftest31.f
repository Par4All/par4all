      program hpftest31
      real a(10,10), b(10,10)
      integer i,j
chpf$ template t(10,10)
chpf$ align a(i,j), b(i,j) with t(i,j)
chpf$ processors p(2)
chpf$ distribute t(block, *) onto p
      print *, 'hpftest31 running'
chpf$ independent(i,j)
      do i=1, 10
         do j=1, 10
            a(i,j)=(i-1)*10+j-1
         enddo
      enddo
chpf$ independent(i,j)
      do i=2, 10
         do j=1, 10
            b(i,j)=a(i-1,j)
         enddo
      enddo
      do i=2, 10
         print *, 'b(',i,',1)=',b(i,1)
      enddo
      print *, 'hpftest31 ended'
      end
