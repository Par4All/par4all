      program chaine03
      character*3 s1, s2
!
      s1 = 'hi'
      s2 = 'hi '
      print *, '-', s1, '-', s2, '-'
      if (s1.eq.s2) then
         print *, 's1 == s2'
      endif
      if (s1.ne.s2) then
         print *, 's1 != s2'
      endif
      end
