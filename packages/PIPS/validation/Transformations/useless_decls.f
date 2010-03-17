! testing removal of useless declarations
      subroutine useless(x1)
! some useless declarations
      integer i, n
      parameter(n=10)
      real a, b(n)
      common /bla/ x, y, z
      real x, y, z
      complex notcalled
      external notcalled
! some useful declarations
      real x1 
      integer j, m
      parameter(m=10)
      real c, d(m)
      common /foo/ v, w
      real v, w
      complex called
      external called
! some code
      c = 3.0
      do j=1, 3
         d(j) = 8
      enddo
      v = called(2)      
      end
