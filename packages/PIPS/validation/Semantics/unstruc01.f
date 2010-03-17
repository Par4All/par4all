      program unstruc01

C     Show bug on non-exiting unstructured by Nga Nguyen

      i = 0

 100  continue
      i = i + 1
c      if(i.eq.n) stop
      if(i.ge.n) stop
      print *, i
      go to 100

      print *, i

      end
