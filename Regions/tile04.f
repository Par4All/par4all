C     Test if the region of a 1-D tile is precise

      subroutine tile04(n, ts, a)
      integer n, ts, a(n)
      integer ti, i
      DO ti=0, n, ts
         DO i = ti, min(ti+ts, n), 1
            a(i)=0
         enddo
      enddo
      end
