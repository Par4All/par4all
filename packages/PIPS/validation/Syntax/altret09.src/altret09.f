      program altret09

C     Check link-edit issues: keep the main and process FOO by PIPS

      i = 0

      call foo(*123, i)
      print *, "First standard return from foo"

      call foo(*123, i)
C     This should not happen!
      print *, "Second standard return from foo"

      print *, 'STOP 0 in ALTRET09'
      stop 0

 123  continue
      print *, 'STOP 1 in ALTRET09'
      stop 1

      end
