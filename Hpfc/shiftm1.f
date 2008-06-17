      program shiftm1
      integer i, n
      parameter (n=10)
      real a(n)

!hpf$ processors P(2)
!hpf$ distribute a(block) onto P

!hpf$ independent
      do i=1, n
        a(i) = real(i)
      end do

      do i=n, 4, -1
        a(i) = a(i-3)
      end do

      print *, (a(i),i=1,n)

      end
