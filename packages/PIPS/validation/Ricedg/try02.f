      program try02

C     Check SRU format for dependence graph with non-uniform dependences

      real x(10,10,10)
      real s(10)

      do i = 1, 10
         s(i) = 0.
         do j = 1, 10
            do k = 1, 10
               s(i) = s(i) + x(i,j,k)
            enddo
         enddo
      enddo

      end
