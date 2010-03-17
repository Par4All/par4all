      program hpftest17
      integer i
      real a(100), b(100), s
CHPF$ TEMPLATE t(100)
CHPF$ ALIGN a(I), b(I) WITH t(I)
CHPF$ PROCESSORS p(4)
CHPF$ DISTRIBUTE t(block) ONTO p
      s = 8.0
CHPF$ INDEPENDENT(I)
      do i=1,100
         a(i) = b(i) + s
      enddo
      end
      
