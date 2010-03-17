      program hpftest20
      integer j(10)
CHPF$ TEMPLATE t(10)
CHPF$ ALIGN j(I) with t(I)
CHPF$ PROCESSORS p(5)
CHPF$ DISTRIBUTE t(block) ONTO p
      i = 0
 1000 format(i6)
CHPF$ INDEPENDENT(I)
      do i=1,10
         j(i)=i
      enddo
      do i=1,10
         write(1, 1000) j(i)
      end do
      end
