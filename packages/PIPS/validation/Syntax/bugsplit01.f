!
! cannot stand a continuation in the declaration.
!
      program 
     $     split01

      print *, entry foo()
      call bla
      call funny
      end

      function entry foo
!
! cannot stand a continuation with an entry-looking constant
!
      entry foo
     $     = 2.1
      end

      subroutine bla
      integer entry bil(10)
!
! cannot stand an entry-looking array, even without continuation
!
      entry bil(1) = 1
      end

c      subroutine bli
c      integer entry bli(10)
c!
c! cannot stand an entry-looking array, even without continuation
c! especially with an already existing module...
c!
c      entry bli(1) = 1
c      end

!
! cannot stand a continuation in the declaration, again
!
      subroutine fun
     $     ny
      print *, 'funny'
      end
