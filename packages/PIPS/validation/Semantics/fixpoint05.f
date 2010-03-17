      program fixpoint05

C     Suppose J is known to be greater than 5 (in fact, equal to 5
C     because of the previous loop) and the loop certainly entered: the
C     information is lost!

C     The problem seems to be due to the precondition equation for
C     entered loops: p' = t(t*(p)). The certainly executed iteration is
C     applied too late, when the information brought by p is lost thru
C     t*. However, t*(t(p)) would still not be satisfactory. t* should
C     be computed for the subset defined by p.

      if(j.ge.5) then

         do i = 1, 5
            if(j.ge.5) then
               j = 5
            endif
         enddo

         print *, j

      endif

      end
