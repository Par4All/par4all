      program type09

C     Check float behavior after extensions to non-integer types

      read *, r

      if (r.gt.0.0.and.r.lt.1.0) then
         print *, r
         print *, r
c      else
c         print *, r
c         print *, r
      endif

c      if (r.gt.0.0.or.r.lt.1.0) then
c         print *, r
c         print *, r
c      else
c         print *, r
c         print *, r
c      endif

      end
