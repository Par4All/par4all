      program tell
      integer i
      real a(10)
!hpf$ processors p(2)
!hpf$ distribute a(block) onto p

!fcd$ tell('tell one')
      do i=1, 10
        a(i) = i*0.333
      enddo
!fcd$ tell('tell two')
      end
