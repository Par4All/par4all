      program hpftest16
      integer i,j
      real a(100,100), b(100,100), c(100,100), temp
CHPF$ TEMPLATE t(100,100)
CHPF$ ALIGN a(I,J), b(i,J), c(I,J) WITH t(I,J)
CHPF$ PROCESSORS p(4,4)
CHPF$ DISTRIBUTE t(block, block) ONTO p
      temp = 0
CHPF$ INDEPENDENT(I,J)
      do i=1,100
         do j=1,100
            a(i,j) = b(i,j) + c(i,j)
         enddo
      enddo
      end
      
