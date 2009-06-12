! The goal of this test case is to check that all basic cases where
! reduction is not possible are well detected by pips. None of the loops in
! the programm should be reduced

      program parallel
      do i = 1, n
         s = 1 - s
      end do
      end
