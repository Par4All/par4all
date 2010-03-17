      program type10

C     Check integer behavior after extensions to non-integer types

      read *, i

      if (i.gt.0.and.i.lt.1) then
         print *, i
c      else
c         print *, i
      endif

c      if (i.gt.0.or.i.lt.1) then
c         print *, i
c      else
c         print *, i
c      endif

      end
