      program type07

C     Goal: check the extension of the semantics analysis to lfoat
C     scalar variables in tests

      real x

      x = 0.57

      if(x.ge.0.58) then
         read *, x
         print *, x
      else
         read *, x
         print *, x
      endif

      end
