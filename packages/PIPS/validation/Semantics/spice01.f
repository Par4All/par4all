      program spice01

C     Aliasing between a global variable and a formal parameter

      common /foo/ maxmem

      call setmem(maxmem)

      end

      subroutine setmem(ksize)

      common /foo/ maxmem

C     This is not smart but programmed in Perfect's spice.f

      maxmem = ksize

      end
