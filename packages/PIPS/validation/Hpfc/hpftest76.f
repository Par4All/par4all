      program hpftest76
      integer j(10)
CHPF$ TEMPLATE t(10)
CHPF$ ALIGN j(I) with t(I)
CHPF$ PROCESSORS p(5)
CHPF$ DISTRIBUTE t(block) ONTO p
 1000 format(i6)
      j(1)=3
      write(1, 1000) j(1)
      end
