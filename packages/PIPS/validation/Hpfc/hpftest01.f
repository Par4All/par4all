      program hpftest01
      real a(10)
CHPF$ TEMPLATE t(10)
CHPF$ ALIGN a(I) with t(I)
CHPF$ PROCESSORS p(2)
CHPF$ DISTRIBUTE t(block) ONTO p
      i=0
CHPF$ INDEPENDENT(i)
      do i=1,10
         a(i)=i
      enddo
      end
