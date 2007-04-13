      subroutine induc03(a, n)

C     Example submitted by Nga. You want to prove that ij is greater
C     than 1. Try to compute proper transformer the first time (see
C     induc02.f). Final print added to obtain an additional block

      real a(n)

      do ix = 1, nx2
         ij = ix
         do iy = 1, ny2
               a(ij) = 0.
               ij = ij + nx2
         enddo
      enddo
      print *, ix

      end
