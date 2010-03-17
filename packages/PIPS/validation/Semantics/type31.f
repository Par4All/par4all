      program type31

C     Check that string values are not used when strings are not
C     analyzed even when they are stored in integer variable

      data kk /'('/

      if(kk .eq. ')') then
C     Is kk's value known although strings are not analyzed?
         print *, kk
      endif

C     This assignment is not ANSI compatible
      kk = '?'

      print *, kk

      end
