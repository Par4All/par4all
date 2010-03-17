! ENTRY13
!
! a subroutine with entries and a global counter
!
      program entry13
      print *, 'entry13'
      call bla1
      call bla2
      call bla3
      call bla2
      call bla3
      call bla1
      call blax
      end
! BLA
      subroutine blax
      integer i
! warning: this initialization should appear only once...
      data i /0/
! data => save i
      print *, 'BLAX:I = ', i
      return
      entry bla3
      i = i + 1
      entry bla2
      i = i + 1
      entry bla1
      i = i + 1
      end
