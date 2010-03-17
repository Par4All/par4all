C     Declaration bug observed in aile.f: a conflict MAY occur if a local
C     variable has the same name as a function (may be also true for
C     other global variables, subroutines and commons);
C
C     A conflict ALWAYS occurs if a COMMON and a FUNCTION or a SUBROUTINE
C     have the same name
C
C     If function d is parsed BEFORE function e, e cannot be parsed; e.d seems
C     to be identified with top-level:d
C
C     Francois Irigoin, April 91
      program funvar
      x = d(5)
      y = e(6)
      end
      function d(x)
      d = x
      return
      end
      function e(x)
      d = x
      e = d
      return
      end
