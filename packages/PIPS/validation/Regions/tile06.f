C     Test if the region of a 1-D tile is precise

      subroutine tile06(n, ts, a)
      integer n, ts, a(n)
      integer ti, i
      if(n.gt.0.and.ts.gt.0) then
         DO ti=1, n, ts
            DO i = ti, ti+ts, 1
               a(i)=0
            enddo
         enddo
      endif
      end
