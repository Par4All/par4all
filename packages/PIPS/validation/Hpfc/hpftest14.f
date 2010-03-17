      program hpftest14
      integer i,j
      real a(100,100)
CHPF$ TEMPLATE t(100)
CHPF$ ALIGN a(I,*) WITH t(I)
CHPF$ PROCESSORS p(4)
CHPF$ DISTRIBUTE t(block) ONTO p
      i = 0
CHPF$ INDEPENDENT(I,J)
      do i=2,101
         do j=0,99
            a(i-1,j+1)=i+j
         enddo
      enddo
      end
