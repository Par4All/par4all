      program try03

C     Check SRU format for dependence graph with non-uniform dependences

      real s(10)

      do i = 1, 5
         s(2*i) = s(i)
      enddo

      end
