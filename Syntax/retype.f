C     Test of type redefinitions and their impact on allocations in commons
      program retype
      common /foo/ x, i, y, z, u, v
      integer x
      real i
      logical y
      complex z
      real*8 u
      complex*16 v

      x = 1

      call retype2
      call retype3

      end

      subroutine retype2
      common /foo/ x, i, y, z, u, v
      integer x
      real i
      logical y
      complex z
      real*8 u
      complex*16 v

      x = 2

      end

      subroutine retype3
      common /foo/ x, i, y, z, u, v
      integer x
      real i
      logical y
      complex z
      real*8 u
      complex*16 v

      x = 3

      end
