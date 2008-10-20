      program hpftest03
      real a1(10),a2(10,10)
      integer j1(100),j2(4:10,10)
Ce sont les directives HPF:
C
CHPF$ TEMPLATE t1(10),t2(100,100)
C
CHPF$ ALIGN a1(I) with t1(I)
CHPF$ ALIGN a2(I,J) with t2(J,I)
CHPF$ ALIGN j1(*) with t1(4)
CHPF$ ALIGN j2(I,J) with t2(-2*I+60,2*J-1)
C
CHPF$ PROCESSORS p1(2),p2(20,10),p3
C     
CHPF$ DISTRIBUTE t1(cyclic(3)) ONTO p1
CHPF$ DISTRIBUTE t2(block(10),cyclic) onto p2
C
Commencement du code
      do i=1,10
         a1(i)=i
      enddo
      end
