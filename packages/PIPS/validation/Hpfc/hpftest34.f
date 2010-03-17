      program hpftest34
      real a(10,10), b(10,10)
      integer i,j
chpf$ template t(10,10)
chpf$ align a(i,j), b(i,j) with t(i,j)
chpf$ processors p(2,2)
chpf$ distribute t(block,block) onto p
      print *, 'hpftest34 running'
chpf$ independent(i,j)
      do i=1, 10
         do j=1, 10
            a(i,j)=10*i+j-11
         enddo
      enddo
chpf$ independent(i,j)
      do i=2, 9
         do j=2, 9
            b(i,j)=a(i-1,j)+a(i-1,j-1)+a(i-1,j+1)+a(i,j)
         enddo
      enddo
      do i=2, 9
         print *, 'b(',i,',  5)=',b(i,5)
         print *, 'b(  6,',i,')=',b(6,i)
      enddo
      print *, 'hpftest34 ended'
      end
