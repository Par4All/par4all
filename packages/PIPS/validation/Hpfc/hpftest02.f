      program hpftest02
      real a1(10),a2(11,12),a3(10,10),a4(10,10)
Ce sont les directives HPF:
C
CHPF$ TEMPLATE t1(10),t2(20,30)
C
CHPF$ ALIGN a1(I) with t1(I)
CHPF$ ALIGN a2(I,J) with t2(J,I)
CHPF$ ALIGN a3(I,*) with t2(*,I+5)
CHPF$ ALIGN a4(I,J) with t2(I,2*J+3)
C
CHPF$ PROCESSORS p1(2),p2(20,10),p3
CHPF$ PROCESSORS p4(14)
C     
CHPF$ DISTRIBUTE t1(block) ONTO p1
CHPF$ DISTRIBUTE t2(block(10),cyclic) onto p2
C
Commencement du code
      do i=1,10
         a1(i)=i
      enddo
      end
