      program hpftest19
      integer j(10), d
CHPF$ TEMPLATE t(10)
CHPF$ ALIGN j(I) with t(I)
CHPF$ PROCESSORS p(3)
CHPF$ DISTRIBUTE t(block) ONTO p
      print *, 'hpftest19 running'
 1000 format(i6)
      print *, 'please enter base'
      read(5, 1000) d
CHPF$ INDEPENDENT(I)
      do i=1,10
         j(i) = d + i
      end do
      do i=1,10
         print *, 'j(',i,') = ',j(i)
      end do
      print *, 'hpftest19 ended'
      end
