      program process(sig)
      integer f, n
      parameter(n=32768)
      real sig(n)
      real spectre(n)

      call fft1(sig, spectre)

      p = 0.
      do f = 1, n
         p = p + spectre(f)**2
      enddo
      end


      subroutine fft1(s, f)
      integer i, n
      parameter(n=32768)
      real s(n), f(n)

      do i = 1, n
         f(i) = s(i)
      enddo

      end
