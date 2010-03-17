      program hpftest09
      integer g(10)
      integer j,k,l,m,n
CHPF$ TEMPLATE t(10)
CHPF$ ALIGN g(I) with t(I)
CHPF$ PROCESSORS p(2)
CHPF$ DISTRIBUTE t(block) ONTO p
      k=1
      j=g(k)+1
      l=g(g(j)+2)+3
      m=j+k+l
      n=0
      end
