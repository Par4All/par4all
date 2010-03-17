! very simple reduction
      program reduction
      integer n, i
      parameter(n=10)
      real a(n), s
!hpf$ processors P(4)
!hpf$ distribute a(block) onto P
!hpf$ independent
      do i=1, n
         a(i)=real(i)
      enddo
      s=0.33
!hpf$ independent, reduction(s)
      do i=2, n-1
         s = s * a(i)
      enddo
      print *, 'result is (should be near 119750.398)', s
      end
