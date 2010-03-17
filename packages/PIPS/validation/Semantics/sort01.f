      program sort01

C     Check that proper values are retained from equations for
C     substitution in inequalities

      if(i.lt.3) then
C     you are not interested in i#init
         i = i + 1
         print *, i
      endif

c      j = i
c      if(j.lt.3) then
C     You want the information on i
c         print *, i
c      endif

      end
