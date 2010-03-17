      program hpftest05
      real a(10)
      integer j
CHPF$ TEMPLATE t(10)
CHPF$ ALIGN a(I) with t(I)
CHPF$ PROCESSORS p(2)
CHPF$ DISTRIBUTE t(block) ONTO p
      j=1
CHPF$ INDEPENDENT(I)
      do i=j,10
         a(i)=i
      enddo
      end
