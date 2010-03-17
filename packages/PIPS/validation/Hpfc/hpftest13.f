      program hpftest13
      integer i
      real a(100)
CHPF$ TEMPLATE t(100)
CHPF$ ALIGN a(I) WITH t(I)
CHPF$ PROCESSORS p(4)
CHPF$ DISTRIBUTE t(block) ONTO p
      i = 0
CHPF$ INDEPENDENT(I)
      do i=1,100
         a(i)=i
      enddo
      end
