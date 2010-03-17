      program common_size01

C     Check that common instances with different sizes are detected
C     and that common instances occuring before typing information
C     is available do not generate a warning

C     Common declared before its members are typed
c      common /size_ok/mot, x, i
      common /size_ok/mot
      character*80 mot
      dimension x(10)

C     Varying size common
      common /size_not_ok/ t(10)

      call reorder

      end

      subroutine reorder

C     Common declared after its members are typed
      character*80 mot
      dimension x(10)
c      common /size_ok/mot, x, i
      common /size_ok/mot

C     Varying size common
      common /size_not_ok/ t(20)

      print *, i

      end
