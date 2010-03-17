! regeneration of included commons...
      program inclusions
! some commons
      common /foo/ a, b, c
      common /bla/ x, y, z
      common // q, w, t
      common /equiv/ i, j, k
      integer l(3)
      equivalence (i, l(1))
      common /diff/ d1, d2, d3
      common /diffe/ e1, e2, e3
      integer e(3)
      equivalence (e1, e(1))
! some code
      a = 1
      x = 2
      q = 3
      i = 4
      d1 = 5
      e1 = 6
      call pfoo
      call pfoobla
      call pblank
      call pequiv
      call pdiff
      call pdiffe
      end

      subroutine pfoo
      common /foo/ a, b, c
      print *, a, b, c
      end

      subroutine pfoobla
      common /foo/ a, b, c
      common /bla/ x, y, z
      print *, a, b, c
      print *, x, y, z
      end

      subroutine pblank
      common // q, w, t
      print *, q, w, t
      end

      subroutine pequiv
      common /equiv/ i, j, k
      integer l(3)
      equivalence (i, l(1))
      print *, i, j, k
      end

      subroutine pdiff
      common /diff/ d1, d2, dd
      print *, d1, d2, dd
      end

      subroutine pdiffe
      common /diffe/ e1, e2, ee
      integer e(3)
      equivalence (e1, e(1))
      print *, e1, e2, ee
      end
