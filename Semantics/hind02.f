      program hind02

c     PLDI'95, Constant propagation, Figure 6

c     The values of f1, f2, f3, f4, f5 and f6 are found as well as the
c     values of sub1 and sub2. However, the value of sub1 is not
c     returned to hind02 because the transformer of sub1 is used
c     instead. The value of sub2 is returned, because the transformer of
c     sub2 is exact.

      integer r1, sub1

      r1 = sub1(0)

      if(r1.eq.0) then
C     See if this is reachable
         call sub4()
      endif

      end

      integer function sub1(f1)
      integer f1
      integer x, y, r2
      integer sub2
      
      x = 9

      if(f1.eq.0) then
         y = 1
      else
         y = 0
      endif

      r2 = sub2(y, 4, f1, x)

      call sub3(r2)

      if(r2.gt.0) then
         sub1 = 1
      else
         sub1 = 0
      endif

      end

      integer function sub2(f2, f3, f4, f5)
      integer f2, f3, f4, f5

      sub2 = f2+f3+f4+f5

      end

      subroutine sub3(f6)
      integer f6
      print *, f6
      end

      subroutine sub4
      print *, 'sub4 is called'
      end
