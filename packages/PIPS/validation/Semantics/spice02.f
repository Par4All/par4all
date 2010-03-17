      program spice02

C     Aliasing between teo formal parameters

      call setmem(maxmem, maxmem)

      print *, maxmem

      end

      subroutine setmem(ksize, maxsize)

C     This is not smart but could be programmed

      ksize = maxsize

      end
