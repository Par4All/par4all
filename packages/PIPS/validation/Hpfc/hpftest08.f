      program hpftest08
      integer g(10)
      integer j
CHPF$ TEMPLATE t(10)
CHPF$ ALIGN g(I) with t(I)
CHPF$ PROCESSORS p(2)
CHPF$ DISTRIBUTE t(block) ONTO p
      j=1
      j=g(1)
      end
