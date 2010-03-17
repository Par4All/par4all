      program chaine05

C     Fortran does not take trailing SPACEs into account when comparing
C     strings because any string is assumed padded with SPACE up to its
C     declared length

      character*10 s
      character*10 t
      character*10 u
      character*20 v
      character*20 w

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
!        TRUE because s and t are declared with the same length and hence padding
         print *, "Trailing blanks are not ignored when concatenating"
      endif

      print *, s, t, u, v, w

      end
