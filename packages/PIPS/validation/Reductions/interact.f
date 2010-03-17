      program interact
      s = 0.
      p = 1.
      call sumprod(s, p, 2.1)
      call sumprod(s, p, 2.+i)
      do i=1, n
         call sumprod(s, p, 2.+i)
         call sumprod(s, p, fsumprod(s, p, 3.))
      enddo
      do i=1, n
         call sumprod(p, s, fsumprod(s, p, 3.))
      enddo
      do i=1, n
         call sumprod(s, p, 2.+i)
         call sumprod(p, s, 1.-i)
      enddo
      end
      subroutine sumprod(x, y, z)
      x = x + z
      y = y * z
      end
      real function fsumprod(x, y, z)
      x = x + z
      y = y * z
      fsumprod = z
      end
!tps$ activate PRINT_CODE_PROPER_REDUCTIONS
!tps$ display PRINTED_FILE($ALL)
!tps$ activate PRINT_CODE_CUMULATED_REDUCTIONS
!tps$ display PRINTED_FILE($ALL)
