      program type16
      real a, b

      read *, a

      a = amax1(a+1., 4., 5.)

      b = 3.

      if(a.lt.b-0.5) then
         print *, 'a is small'
      else
         print *, 'a is big'
      endif

      end
