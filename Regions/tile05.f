C     Test if the region of a 1-D tile is precise

C     Part of tests tile01, tile02, tile03

      subroutine tile05(n, ts, a)
      integer n, ts, a(n)
      integer ti, i, m

C     if(.true.) does not disturb exact region computation
      if(.true.) then
         a(1) = 0
      endif

      do ti=0, n, ts
         if(.true.) then
C           Must be a case of *exact* transformer that Beatrice would
C           like to capture...
            m = min(ti+ts, n)
            DO i = ti, m, 1
               a(i)=0
            enddo
         endif
      enddo
      end
