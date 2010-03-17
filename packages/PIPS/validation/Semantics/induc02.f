      subroutine induc02(a, n)

C     Example submitted by Nga. You want to prove that ij is greater
C     than 1 in the second loop. a new property was added to semantics
C     to recompute the loop fixpoints with precondition information.

      real a(n)

      do ix = 1, nx2
         ij = ix
         do iy = 1, ny2
            if(nx2.gt.0) then
               a(ij) = 0.
               ij = ij + nx2
            endif
         enddo
      enddo

      do ix = 1, nx2
         ij = ix
         do iy = 1, ny2
               a(ij) = 0.
               ij = ij + nx2
         enddo
      enddo

      end
