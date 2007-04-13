!
! testing unsplit with one file
!
      program unsplit
      call callee
      call other
      end

      subroutine callee
      print *, 'callee'
      end

      subroutine other
      print *, 'other'
      end
