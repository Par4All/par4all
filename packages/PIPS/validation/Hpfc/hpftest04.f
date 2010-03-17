      program hpftest04
      real a(10),b(10)
Ce sont les directives HPF:
C
CHPF$ TEMPLATE t(10)
C
CHPF$ ALIGN a(I) with t(I)
CHPF$ ALIGN b(I) with t(I)
C
CHPF$ PROCESSORS p(2)
C     
CHPF$ DISTRIBUTE t(block) ONTO p
C
Commencement du code
CHPF$ INDEPENDENT(i)
      do i=1,10
         a(i)=i
      enddo
      do i=1,10
         b(11-i)=a(i)
      enddo
      do i=1,10
         a(i)=b(i)
      enddo
      end
