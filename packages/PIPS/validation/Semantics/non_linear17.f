      subroutine non_linear17

C     For EDF KMAX, refine transformers

      kmax = 1
      call foo(kmax)
      print *, kmax

      end

      subroutine foo(kmax)
      if(kmax.ne.1) then
         kmax = kmax-1
      endif

      end
