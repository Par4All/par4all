      program loopseq

C     Check exit condition from first loop onto second loop

C     Bug: the second loop on I is not properly analyzed.
C     It cannot be shown to be dead code

C     Fixed on June 22, 1997

      real t(100)

      if(n.ge.1) then

         do i = 1, n
            t(i) = 0.
         enddo

C     This loops leads to i <= i <= n and fortunately the
C     left part disappears as a trivial constraint. But it should
C     never have been built in the first place!

C     Bug fixed on June 22, 1997

         do i = i, n
            t(i) = 0.
         enddo

         print *, i

      endif

      end
