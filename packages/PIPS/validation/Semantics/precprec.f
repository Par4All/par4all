      program precprec
C     Check that preconditions are properly evaluated after a
C     partial evaluation

      i = 2
      read *, j
      call assign(i*j,k)
      print *, k

      end

      subroutine assign(m,n)
      n = m
      end
