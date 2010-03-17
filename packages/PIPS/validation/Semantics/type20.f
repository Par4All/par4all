      program type20
      character*3 s1, s2
      integer i
!
      s1 = 'hi '
      s2 = 'hi '
      i  = 0
      print *, '-', s1, '-', s2, '-'
      if (s1.eq.s2) then
         i = i+1
         print *, 's1 == s2'
      endif
      if (s1.ne.s2) then
         i = i+1
         print *, 's1 != s2'
      endif
      if (s1.lt.s2) then
         i = i+1
         print *, 's1 <  s2'
      endif
      if (s1.gt.s2) then
         i = i+1
         print *, 's1 >  s2'
      endif
      if (s1.le.s2) then
         i = i+1
         print *, 's1 <= s2'
      endif
      if (s1.ge.s2) then
         i = i+1
         print *, 's1 >= s2'
      endif
!
      s1 = 'hi'
      s2 = 'hi '
      print *, '-', s1, '-', s2, '-'
      if (s1.eq.s2) then
         i = i+1
         print *, 's1 == s2'
      endif
      if (s1.ne.s2) then
         i = i+1
         print *, 's1 != s2'
      endif
      if (s1.lt.s2) then
         i = i+1
         print *, 's1 <  s2'
      endif
      if (s1.gt.s2) then
         i = i+1
         print *, 's1 >  s2'
      endif
      if (s1.le.s2) then
         i = i+1
         print *, 's1 <= s2'
      endif
      if (s1.ge.s2) then
         i = i+1
         print *, 's1 >= s2'
      endif
!
      s1 = 'bye'
      s2 = 'hi'
      print *, '-', s1, '-', s2, '-'
      if (s1.eq.s2) then
         i = i+1
         print *, 's1 == s2'
      endif
      if (s1.ne.s2) then
         i = i+1
         print *, 's1 != s2'
      endif
      if (s1.lt.s2) then
         i = i+1
         print *, 's1 <  s2'
      endif
      if (s1.gt.s2) then
         i = i+1
         print *, 's1 >  s2'
      endif
      if (s1.le.s2) then
         i = i+1
         print *, 's1 <= s2'
      endif
      if (s1.ge.s2) then
         i = i+1
         print *, 's1 >= s2'
      endif
!
      s1 = 'hi'
      s2 = 'bye'
      print *, '-', s1, '-', s2, '-'
      if (s1.eq.s2) then
         i = i+1
         print *, 's1 == s2'
      endif
      if (s1.ne.s2) then
         i = i+1
         print *, 's1 != s2'
      endif
      if (s1.lt.s2) then
         i = i+1
         print *, 's1 <  s2'
      endif
      if (s1.gt.s2) then
         i = i+1
         print *, 's1 >  s2'
      endif
      if (s1.le.s2) then
         i = i+1
         print *, 's1 <= s2'
      endif
      if (s1.ge.s2) then
         i = i+1
         print *, 's1 >= s2'
      endif
!
      s1 = 'hi world'
      s2 = 'hi'
      print *, '-', s1, '-', s2, '-'
      if (s1.eq.s2) then
         i = i+1
         print *, 's1 == s2'
      endif
      if (s1.ne.s2) then
         i = i+1
         print *, 's1 != s2'
      endif
      if (s1.lt.s2) then
         i = i+1
         print *, 's1 <  s2'
      endif
      if (s1.gt.s2) then
         i = i+1
         print *, 's1 >  s2'
      endif
      if (s1.le.s2) then
         i = i+1
         print *, 's1 <= s2'
      endif
      if (s1.ge.s2) then
         i = i+1
         print *, 's1 >= s2'
      endif
!
      s1 = '   '
      s2 = ' '
      print *, '-', s1, '-', s2, '-'
      if (s1.eq.s2) then
         i = i+1
         print *, 's1 == s2'
      endif
      if (s1.ne.s2) then
         i = i+1
         print *, 's1 != s2'
      endif
      if (s1.lt.s2) then
         i = i+1
         print *, 's1 <  s2'
      endif
      if (s1.gt.s2) then
         i = i+1
         print *, 's1 >  s2'
      endif
      if (s1.le.s2) then
         i = i+1
         print *, 's1 <= s2'
      endif
      if (s1.ge.s2) then
         i = i+1
         print *, 's1 >= s2'
      endif

      print *, i
      end
