      program nga08

C     To address Nga concern about using useless float and string
C     constants when analyzing integer variables

      i = 1
      if(x.gt.1.5) then
         i = 2
      endif
      
      print *,i

      END
      
