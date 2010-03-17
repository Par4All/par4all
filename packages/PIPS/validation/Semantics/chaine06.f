      program chaine06

C     Fortran does not take trailing SPACEs into account when comparing
C     strings because any string is assumed padded with SPACE up to its
C     declared length

      character*10 s
      character*11 t
      character*12 u
      character*21 v
      character*22 w

      s = "hello"
      t = "hello "

      if(s.eq.t) then
!        FALSE
         print *, "Trailing blanks are ignored"
      else
!        TRUE 
         print *, "Trailing blanks are not ignored"
      endif

      s = "hello"
      t = " hello"

      if(s.eq.t) then
!        FALSE
         print *, "Leading blanks are ignored"
      else
!        TRUE 
         print *, "Leading blanks are not ignored"
      endif

      s = "hello"
      t = "hello "
      u = "world"
      v = s // u
      w = t // u

      if(v.eq.w) then
!        FALSE
         print *, "Trailing blanks are ignored when concatenating"
      else
!        TRUE 
         print *, "Trailing blanks are not ignored when concatenating"
      endif

      print *, s
      print *, t
      print *, u
      print *, v
      print *, w

      end
