      program dead1

C     check that precondition-based dead code elimination is effective
C     Used to check that control-based dead code elimination was effective
C     (else gen_multi_recurse() would reach unreachable statements)
C     Should be mdae tougher when Ronan HCFG restructuring is available

      call pr(1)

      i = 1

c      if(i.gt.1) go to 100

c      call pr(2)

c 100  continue

C     No need for dead code elimination here, because the preconditions
C     do the job

      if(i.gt.1) then
         call pr(2)
      endif

      end

      subroutine pr(i)

      print *,i

      end
