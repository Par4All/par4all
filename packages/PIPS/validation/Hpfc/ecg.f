      program ecg
      integer v, n, i
      parameter(n=32768)
      real signal(n,5)
      common /result/ power(5)
      real power 

!hpf$ processors proc(5)
!hpf$ template t(5)
!hpf$ align with t(i):: signal(*,i), power(i)
!hpf$ distribute t(block) onto proc

      external process
      real process
!fcd$ pure process

      do v = 1, 5
         read *, (signal(i, v), i=1, n)
      enddo

!hpf$ independent
      do v = 1, 5
         power(v) = process(signal(1,v))
      enddo

      call printout
      end


      real function process(sig)
!fcd$ pure
!fcd$ pure fft1
      integer f, n
      parameter(n=32768)
      real sig(n)
      real spectre(n)

      call fft1(signal, spectre)

      p = 0.
      do f = 1, n
         p = p + spectre(f)**2
      enddo
      process = sqrt(p)
      end


      subroutine fft1(s, f)
!fcd$ pure
      integer i, n
      parameter(n=32768)
      real s(n), f(n)

      do i = 1, n
         f(i) = s(i)
      enddo

      end

      subroutine printout
      common /result/ power(5)
      real power, p
!hpf$ processors proc(5)
!hpf$ template t(5)
!hpf$ align with t:: power
!hpf$ distribute t(block) onto proc
      integer v

      p = 0.
!hpf$ independent, reduction(p)
      do v = 1, 5
         p = p + power(v)
      enddo

      print *, p

      end
