      program hpftest26
      integer j(10), k(3)
CHPF$ TEMPLATE t(10)
CHPF$ ALIGN j(I), k(*) with t(I)
CHPF$ PROCESSORS p(5)
CHPF$ DISTRIBUTE t(block) ONTO p
      print *, 'hpftest26 running'
      do i=1,3
         k(i) = i
      enddo
chpf$ independent(i)
      do i=1,10
         j(i) = k(mod(i,3)+1) + (10*i)
      enddo
      do i=1,10
         print *,'j(',i,') = ',j(i)
      enddo
      print *, 'hpftest26 ended'
      end
