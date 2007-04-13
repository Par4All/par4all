      program type11

C     Check integer behavior after extensions to non-integer types

      read *, i

! known-in

      i = 0
      r = 0.5
      d = 0.5D0

! int gt lt

      if (i.gt.0) then
         print *, 'int gt int'
         print *, i
      endif

      end
