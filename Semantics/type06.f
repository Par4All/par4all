      program type06

C     Goal: check the extension of the semantics analysis to string
C     scalar variables in tests

      character*4 c1

      c1 = "TITI"

      if(c1.eq."TOTO") then
         read *, c1
         print *, c1
      endif

      end
