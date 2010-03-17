      program type15
      real a, b

      read *, a

      a = amax1(a+1., 4., 5.)

      if(a.lt.3.) then
         print *, 'impossible'
      else
         print *, 'of course'
      endif

      end
