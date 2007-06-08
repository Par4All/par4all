      subroutine sortpol(l,m,n)

c     Check that polynomes and monomes are printed in an
c     orderly fashion

      real t(100), u(100,100,100)

c     check for linear complexity, expected: l + m + n + ...
      if(n.gt.m) then
         do i = 1, l
            t(i) = 0.
         enddo
         do k = 1, n
            t(k) = 0.
         enddo
         do j = 1, m
            t(j) = 0.
         enddo
      endif

c     check for global correctness for all thrid degree terms, expected:
c     l^3 + l^2*m + l^2*n +
      if(n.gt.m) then
c     check for monome sort, expected: l*m*n
         do i = 1, l
            do k = 1, n
               do j = 1, m
                  u(i,j,k) = 0.
               enddo
            enddo
         enddo
c     three power of 3
         do i = 1, l
            do k = 1, l
               do j = 1, l
                  u(i,j,k) = 0.
               enddo
            enddo
         enddo
         do i = 1, n
            do k = 1, n
               do j = 1, n
                  u(i,j,k) = 0.
               enddo
            enddo
         enddo
         do i = 1, m
            do k = 1, m
               do j = 1, m
                  u(i,j,k) = 0.
               enddo
            enddo
         enddo

c     six third degree terms with one power of 2

         do i = 1, l
            do k = 1, l
               do j = 1, m
                  u(i,j,k) = 0.
               enddo
            enddo
         enddo
         do i = 1, l
            do k = 1, n
               do j = 1, l
                  u(i,j,k) = 0.
               enddo
            enddo
         enddo

         do i = 1, n
            do k = 1, n
               do j = 1, m
                  u(i,j,k) = 0.
               enddo
            enddo
         enddo
         do i = 1, l
            do k = 1, n
               do j = 1, n
                  u(i,j,k) = 0.
               enddo
            enddo
         enddo

         do i = 1, m
            do k = 1, n
               do j = 1, m
                  u(i,j,k) = 0.
               enddo
            enddo
         enddo
         do i = 1, l
            do k = 1, m
               do j = 1, m
                  u(i,j,k) = 0.
               enddo
            enddo
         enddo

      endif

      end
