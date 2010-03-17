C     interprocedural translation: complex <-> real
      program TCOMP
      real RA(10), RAB(5,2), RB(5,5,5), RC(10,5,5)

      call TCOMP1(RA)
      call TCOMP1B(RAB)
      
      call TCOMP2(RB(1,3,1), 25)
      call TCOMP2(RB(2,1,1), 25)
      call TCOMP2(RC(1,3,1), 25)
      call TCOMP2(RC, 25)

      print *, RA
      print *, RAB
      print *, RB
      print *, RC
      end

      subroutine TCOMP1(C)
      complex C(5)

      do i = 1, 5
         C(i) = CMPLX(0.0, 0.0)
      enddo            
      end

      subroutine TCOMP1B(C)
      complex C(5)

      do i = 1, 5
         C(i) = CMPLX(0.0, 0.0)
      enddo            
      end

      subroutine TCOMP2(C, N)
      complex C(5,*)

      do i = 1, 5
         do j = 1, N
         C(i,j) = CMPLX(0.0, 0.0)
         enddo
      enddo            
      end
