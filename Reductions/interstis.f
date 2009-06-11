      program interstis
      call init
      do i=1, 10
         a = 1/i
         call reduce(a)
      enddo
      call printred
      end
      subroutine reduce(a)
      common /reduced/ s, p, x, n
      s = s + a
      p = p * a
      x = MAX(x, a)
      n = MIN(n, a)
      end
      subroutine init
      common /reduced/ s, p, x, n
      s = 0.
      p = 1.
      x = -1000.
      n = 1000.
      end
      subroutine printred
      common /reduced/ s, p, x, n
      print *, 's=', s, ' p=', p, ' x=', x, ' n=', n
      end
