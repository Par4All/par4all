      program hpftest10
      integer g(10)
      integer j,k
CHPF$ TEMPLATE t(10)
CHPF$ ALIGN g(I) with t(I)
CHPF$ PROCESSORS p(2)
CHPF$ DISTRIBUTE t(block) ONTO p
      j=2
      k=g(g(1))
      g(1)=1
      g(2)=2
      if (g(2).EQ.g(1)) g(g(1))=1000
      if (j.EQ.1) j=1000
      end
