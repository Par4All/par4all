      program hpftest12
      integer i,j,k
      real a(100,100,100)
CHPF$ TEMPLATE t(100,100)
CHPF$ ALIGN a(I,*,*) with t(I,*)
CHPF$ PROCESSORS p(4,4)
CHPF$ DISTRIBUTE t(block,block) ONTO p
      do i=1,100
         do j=1,100
            do k=1,100
               a(i,j,k)=i+j+k
            enddo
         enddo
      enddo
      end
