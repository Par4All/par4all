C     How about transformer on common variables?
      program cbl
      common /foo/ k
      k = 1
      call incr
      print *, k
      end
      subroutine incr
      common /foo/ k
      k = k + 1
      end
