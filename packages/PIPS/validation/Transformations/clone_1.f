! trying clone transformations...
      program c1
      integer i
! clone on a constant
      call clonee(1)
! comments
! again
! and more
      call clonee(2)
! cannot clone there
      do i=1, 5
         call clonee(i)
      enddo
! check reuse of corresponding clonee
      call clonee(2)
! clone with preconditions
      i = 1
      call clonee(i)
      i = 3
      call clonee(i)
      end

      subroutine clonee(i)
      integer i
      print *, i
      end
