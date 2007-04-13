      program type08

C     Check integer behavior after extensions to other types

c      common /foo/ k
      k = 1
c      call incr
      call incr(k)
      print *, k
      end
      subroutine incr(k)
c      subroutine incr
c      common /foo/ k
      k = k + 1
      end
