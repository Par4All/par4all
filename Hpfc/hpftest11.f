      program hpftest11
      integer g(10)
      integer j,k
CHPF$ TEMPLATE t(10)
CHPF$ ALIGN g(I) with t(I)
CHPF$ PROCESSORS p(2)
CHPF$ DISTRIBUTE t(block) ONTO p
      g(1)=1
      k=g(1)
      end
