! computing pi
      program pi
      integer i, n
      parameter(n=1000)
      real*8 v(n), x, gsum, w
!hpf$ distribute v(block)
      w = 1.0/real(n)
      gsum = 0.0
!fcd$ time
!hpf$ independent, new(x), reduction(gsum)
      do i=1, n
         x = (i-0.5)*w
         v(i)=4.0/(1.0+x*x)
         gsum=gsum+v(i)
      enddo
!fcd$ end time('pi computation')
      print *, 'computed pi value is ', gsum*w
      end
