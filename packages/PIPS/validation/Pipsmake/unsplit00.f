! 1
      program one
      integer i, three
      external three
      print *, 'one'
      call two
      i = tree()
      call four
      call seven
      end
! 2
      subroutine two
      print *, 'two'
      end
! 3
      integer function three
      print *, 'three'
      three = 3
      end
! 4
      subroutine four
      print *, 'four'
      call five
      end
! 5
      subroutine five
      print *, 'five'
      end
! 6
      subroutine six
      print *, 'six'
      end
! 7
      subroutine seven
      call six
      print *, 'seven'
      end
