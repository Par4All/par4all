!
! argh... :-(
!
! comparing a fixed length string seems to ignores trailing spaces:-(
!
      program chaine01
      character*10 s
      character*3 t

      s = 'hello'
      print *, '-', s, '-'
      if (s.eq.'hell') then
! FALSE         
         print *, 's == "hello"'
      endif
      if (s.eq.'hello') then
! TRUE
         print *, 's == "hello"'
      endif
      if (s.eq.'hello   ') then
! TRUE...
         print *, 's == "hello   "'
      endif
      if (s.eq.'hello     ') then
! TRUE...
         print *, 's == "hello     "'
      endif
      if (s.eq.'hello          ') then
! TRUE...
         print *, 's == "hello          "'
      endif
      if (s.eq.'hello          toto') then
! FALSE
         print *, 's == "hello          toto"'
      endif

! ending d is dropped
      s = 'hello world'
      print *, '-', s, '-'

      s = 'hi'
      t = 'hi'
      print *, '-', s, '-', t, '-'
      if (s.eq.t) then
! TRUE
         print *, 's == t == "hi"'
      endif

      t = 'hi '
      print *, '-', s, '-', t, '-'
      if (s.eq.t) then
! TRUE...
         print *, 's == t == "hi "'
      endif

      t = 'hi world'
      print *, '-', s, '-', t, '-'
      if (s.eq.t) then
! TRUE...
         print *, 's == t == "hi " (2)'
      endif

      s = 'hi world'
      t = 'hi '
      print *, '-', s, '-', t, '-'
      if (s.eq.t) then
! FALSE
         print *, 's == t == "hi " (3)'
      endif

      s = 'hi        '
      t = 'hi '
      print *, '-', s, '-', t, '-'
      if (s.eq.t) then
! TRUE...
         print *, 's == t == "hi " (4)'
      endif      
      end
