      program callee

      call p(n)

      call p(m)

      call p(10)

      call p(100)

      end

      subroutine p(i)
      real a(10)
      do j = 1, i
         a(i) = 0.
      enddo
      k = k + k + k + k
      end
