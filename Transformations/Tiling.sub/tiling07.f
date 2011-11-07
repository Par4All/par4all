!     Example 4.1 in Xue 2000
!
!     The tiled code by Xue and by Pips give the same correct results,
!     but they are different.

      program tiling07

      real A(-2:9,-1:6)

      do 100 i = -2, 9
         do j = -1, 6
            a(i,j) = 10*(i+2)+j+1
         enddo
 100  continue

      do i = 1, 9
         do j = 1, 5
            a(i,j) = a(i,j-2)+a(i-3,j+1)
         enddo
      enddo

      print *, a(5,2), a(6,3)

!     Reinitialize A

      do i = -2, 9
         do j = -1, 6
            a(i,j) = 10*(i+2)+j+1
         enddo
      enddo

!     Use tiled code from Xue

      do it = 0, 2
         do jt = max(it-1,0), (it+4)/2
            do i = 3*it+1, 3*it+3
               do j = max(-it+2*jt+1,1), min((-it+6*jt+9)/3,5)
                  a(i,j) = a(i,j-2)+a(i-3,j+1)
               enddo
            enddo
         enddo
      enddo

      print *, a(5,2), a(6,3)

      end
