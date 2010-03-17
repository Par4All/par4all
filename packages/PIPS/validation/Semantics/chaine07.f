      program chaine07

C     Fortran does not take trailing SPACEs into account when comparing
C     strings because any string is assumed padded with SPACE up to its
C     declared length: smaller version of chaine05 for debugging

      character*10 s
      character*10 t

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

      end
