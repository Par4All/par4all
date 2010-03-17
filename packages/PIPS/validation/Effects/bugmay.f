      program bugmay

C     To check PIPS handling of a user error and of a user potential error

      k = 0

      call inc(k)

      call condinc(0)

      call inc(0)


      end

      subroutine inc(k)
      k = k + 1
      end

      subroutine condinc(k)
      if(k.ge.0) then
         k = k + 1
      endif
      end
