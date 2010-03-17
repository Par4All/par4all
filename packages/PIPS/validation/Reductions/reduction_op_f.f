! The goal of this test case is to check that all basic reduction operators
! are well detected by pips. All the loops in the programm should be reduced.

      program parallel
      REAL x, y
      INTEGER j, k
      LOGICAL m, o
      do i = 1, n
         s = s + 1
      end do
      do i = 1, n
         s = s - 1
      end do
      do i = 1, n
         s = s * 2
      end do
      do i = 1, n
         m = m.AND.o
      end do
      do i = 1, n
         m = m.OR.o
      end do
      do i = 1, n
         m = m.EQV.o
      end do
      do i = 1, n
         m = m.NEQV.o
      end do
      do i = 1, n
         y = MAX(y,x)
      end do
      do i = 1, n
         y = MIN(y,x)
      end do
!     do i = 1, n
!     k = IAND(k,j)
!     end do
!     do i = 1, n
!     k = IOR(k,j)
!     end do
!     do i = 1, n
!     k = IEOR(k,j)
!     end do
      end

