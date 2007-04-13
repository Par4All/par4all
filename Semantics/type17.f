      program type17
      character*4 s1, s2

      s1 = "toto"
      s2 = "titi"

      if(s1.eq.s2) then
         print *, 's1 equals s2'
      else
         print *, 's1 does not equal s2'
      endif

      if(s1.lt.s2) then
         i = 0
         print *, 's1 is less than s2'
      else
         i = 1
         print *, 's1 is greater than or equal to s2'
      endif

      if(s1.le.s2) then
         i = 0
         print *, 's1 is less than or equal to s2'
      else
         i = 1
         print *, 's1 is greater than s2'
      endif

      end
