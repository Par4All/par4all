C     Bug: the first loop is not parallelized because N is not privatized

      program loop
      real t(100)

      do i = 1, m
         n = i
         t(n) = 0.
      enddo

      do n = 1, m
         t(n) = 1.
      enddo

      end
