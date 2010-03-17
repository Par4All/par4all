      program cond01

      real x(10)

      if(i.gt.n .and. x(i).gt.z) then
         print *, i
      else
         print *, i
      endif

      if(i.gt.n .or. x(i).gt.z) then
         print *, i
      else
         print *, i
      endif

      end
